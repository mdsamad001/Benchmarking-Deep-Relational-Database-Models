import os
import time
from pathlib import Path

import strictfire
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error

from rt.data import RelationalDataset
from rt.model import RelationalTransformer
from rt.tasks import all_tasks
from rt.hop_control import HopConfig, apply_hop_control


REG_TABLES = {
    "item-sales","user-ltv","item-ltv","post-votes","site-success",
    "study-adverse","user-attendance","driver-position","ad-ctr",
}


def _parse_fanout(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"none","null"}:
        return None
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1].strip()
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _task_tuple(dataset: str, task: str):
    for item in all_tasks:
        db_name, table_name, target_col, leakage_cols = item
        if db_name == dataset and table_name == task:
            return item
    raise ValueError(f"Task not found: {dataset}/{task}")


def _save_ckpt(net, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)


def _load_ckpt(net, path: Path):
    net.load_state_dict(torch.load(path, map_location="cpu"))


@torch.inference_mode()
def _evaluate(net, loader, device, hop_cfg, split_name, table_name, max_eval_steps):
    net.eval()
    task_type = "reg" if table_name in REG_TABLES else "clf"

    preds  = []
    labels = []
    losses = []

    total = min(max_eval_steps, len(loader))
    pbar  = tqdm(total=total, desc=split_name, leave=False)
    it    = iter(loader)

    for _ in range(total):
        batch   = next(it)
        true_bs = int(batch.pop("true_batch_size"))

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)

        batch["masks"][true_bs:, :]      = False
        batch["is_targets"][true_bs:, :] = False
        batch["is_padding"][true_bs:, :] = True

        apply_hop_control(batch, hop_cfg)

        _, yhat = net(batch)

        if task_type == "clf":
            yhat_ = yhat["boolean"][batch["is_targets"]].flatten()
            y_    = batch["boolean_values"][batch["is_targets"]].flatten()
        else:
            yhat_ = yhat["number"][batch["is_targets"]].flatten()
            y_    = batch["number_values"][batch["is_targets"]].flatten()

        if task_type == "reg":
            loss_val = F.mse_loss(yhat_.float(), y_.float()).item()
        else:
            loss_val = F.binary_cross_entropy_with_logits(
                yhat_.float(), (y_.float() > 0).float()
            ).item()

        preds.append(yhat_.detach().float().cpu())
        labels.append(y_.detach().float().cpu())
        losses.append(loss_val)
        pbar.update(1)

    pbar.close()

    preds    = torch.cat(preds).numpy()
    labels   = torch.cat(labels).numpy()
    avg_loss = sum(losses) / max(len(losses), 1)

    if task_type == "reg":
        mse    = mean_squared_error(labels, preds)
        return avg_loss, mse, f"mse={mse:.6f}"
    else:
        lab = (labels > 0).astype(int)
        auc = roc_auc_score(lab, preds)
        return avg_loss, auc, f"auc={auc:.6f}"


def run(
    dataset="rel-stack",
    task="post-votes",
    float_ckpt_path=None,
    epochs=2,
    train_steps_per_epoch=30,
    max_eval_steps=5,
    seq_len=512,
    num_blocks=12,
    d_ff=512,
    d_model=256,
    num_heads=8,
    d_text=384,
    batch_size=256,
    num_workers=4,
    max_bfs_width=64,
    max_hop=2,
    fanout_per_hop="10,10",
    lr=1e-4,
    wd=0.0,
    max_grad_norm=1.0,
):
    os.environ["WANDB_MODE"] = "disabled"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    db_name, table_name, target_col, leakage_cols = _task_tuple(dataset, task)
    task_type = "reg" if table_name in REG_TABLES else "clf"

    hop_list = _parse_fanout(fanout_per_hop)
    hop_cfg  = HopConfig(max_hop=int(max_hop), fanout_per_hop=hop_list, neighbor_pad_value=-1)

    print(f"Task: {db_name}/{table_name}  type={task_type}")
    print(f"epochs={epochs}  train_steps_per_epoch={train_steps_per_epoch}  max_eval_steps={max_eval_steps}")
    print(f"seq_len={seq_len}  blocks={num_blocks}  d_ff={d_ff}  batch={batch_size}")
    print(f"max_hop={hop_cfg.max_hop}  fanout={hop_cfg.fanout_per_hop}")

    def _make_ds(split):
        return RelationalDataset(
            tasks=[(db_name, table_name, target_col, split, leakage_cols)],
            batch_size=batch_size, seq_len=seq_len, rank=0, world_size=1,
            max_bfs_width=max_bfs_width, embedding_model="all-MiniLM-L12-v2",
            d_text=d_text, seed=0,
        )

    train_ds = _make_ds("train")
    val_ds   = _make_ds("val")
    test_ds  = _make_ds("test")

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=None, num_workers=num_workers, pin_memory=True)

    net = RelationalTransformer(
        num_blocks=num_blocks, d_model=d_model, d_text=d_text,
        num_heads=num_heads, d_ff=d_ff, float_ckpt_path=float_ckpt_path,
    ).to(device).to(torch.bfloat16)

    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, fused=True)

    best_metric = float("inf") if task_type == "reg" else float("-inf")
    best_path   = Path("ckpts_sanity") / f"{db_name}_{table_name}_best.pt"

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        net.train()
        train_loader.dataset.sampler.shuffle_py(epoch)
        it     = iter(train_loader)
        t_loss = 0.0
        t_n    = 0

        for _ in range(train_steps_per_epoch):
            batch   = next(it)
            true_bs = int(batch.pop("true_batch_size"))

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            batch["masks"][true_bs:, :]      = False
            batch["is_targets"][true_bs:, :] = False
            batch["is_padding"][true_bs:, :] = True

            apply_hop_control(batch, hop_cfg)

            loss_pretrain, yhat_dict = net(batch)

            targets_mask = batch["is_targets"]
            if targets_mask.any():
                if task_type == "reg":
                    pred        = yhat_dict["number"][targets_mask].flatten().float()
                    y           = batch["number_values"][targets_mask].flatten().float()
                    task_loss_bp = F.mse_loss(pred, y)
                else:
                    logits      = yhat_dict["boolean"][targets_mask].flatten().float()
                    y           = (batch["boolean_values"][targets_mask].flatten().float() > 0).float()
                    task_loss_bp = F.binary_cross_entropy_with_logits(logits, y)
            else:
                task_loss_bp = loss_pretrain * 0.0

            opt.zero_grad(set_to_none=True)
            task_loss_bp.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()

            t_loss += float(task_loss_bp.detach())
            t_n    += 1

        train_loss = t_loss / max(t_n, 1)

        val_loss, val_metric, val_msg = _evaluate(net, val_loader, device, hop_cfg, "val", table_name, max_eval_steps)

        is_better = (val_metric < best_metric) if task_type == "reg" else (val_metric > best_metric)
        if is_better:
            best_metric = val_metric
            _save_ckpt(net, best_path)

        print(f"epoch={epoch}  train={train_loss:.4f}  val_loss={val_loss:.4f}  val_{val_msg}")

    _load_ckpt(net, best_path)
    test_loss, test_metric, test_msg = _evaluate(net, test_loader, device, hop_cfg, "test", table_name, max_eval_steps)

    print("\n" + "=" * 80)
    print(f"best_val_metric={best_metric:.6f}")
    print(f"TEST (best ckpt): {test_msg}  loss={test_loss:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    strictfire.StrictFire(run)
