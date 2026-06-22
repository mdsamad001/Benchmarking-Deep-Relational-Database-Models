import os
import random
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

from rt.data import RelationalDataset
from rt.model import RelationalTransformer
from rt.hop_control import HopConfig, apply_hop_control, apply_hop_control_from_p2f


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_gather_nd(tensor: torch.Tensor) -> List[torch.Tensor]:
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.long)
    all_sizes  = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_len    = int(max(s.item() for s in all_sizes))
    if tensor.numel() < max_len:
        pad    = torch.zeros((max_len - tensor.numel(),), device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    all_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors, tensor)
    return [t[: int(sz.item())] for t, sz in zip(all_tensors, all_sizes)]


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _is_regression_task(table_name: str) -> bool:
    return table_name in {
        "item-sales","user-ltv","item-ltv","post-votes","site-success",
        "study-adverse","user-attendance","driver-position","ad-ctr",
    }


def _parse_fanout(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    s = str(x).strip()
    if s == "" or s.lower() in {"none","null"}:
        return None
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1].strip()
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _make_hop_cfg(max_hop: int, fanout_list: Optional[List[int]]) -> HopConfig:
    if max_hop == 0:
        return HopConfig(max_hop=0, fanout_per_hop=None, neighbor_pad_value=-1)
    if fanout_list is not None:
        if len(fanout_list) != max_hop:
            raise ValueError(
                f"fanout_per_hop has {len(fanout_list)} values but max_hop={max_hop}. "
                f"Expected exactly {max_hop} values."
            )
        return HopConfig(max_hop=max_hop, fanout_per_hop=fanout_list, neighbor_pad_value=-1)
    if max_hop == 1:
        return HopConfig(max_hop=1, fanout_per_hop=[256], neighbor_pad_value=-1)
    if max_hop == 2:
        return HopConfig(max_hop=2, fanout_per_hop=[512, 256], neighbor_pad_value=-1)
    return HopConfig(max_hop=max_hop, fanout_per_hop=[512] + [256] * (max_hop - 1), neighbor_pad_value=-1)


def _safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _infer_num_classes_from_cache(cache: List[dict]) -> int:
    if not cache:
        return 2
    vals = []
    for b in cache[: min(len(cache), 10)]:
        if "boolean_values" in b and "is_targets" in b:
            y = b["boolean_values"][b["is_targets"]].flatten()
            vals.append(y.detach().cpu())
    if not vals:
        return 2
    y     = torch.cat(vals, dim=0).float()
    uniq  = torch.unique(y)
    if uniq.numel() <= 3:
        return 2
    y_int = torch.round(y).long()
    return int(y_int.max().item() + 1)


def _task_loss_and_metric(
    *,
    task_type: str,
    clf_mode: str,
    num_classes: int,
    yhat_dict: dict,
    batch: dict,
) -> Tuple[float, float, float, str]:
    if task_type == "reg":
        pred = yhat_dict["number"][batch["is_targets"]].flatten().float()
        y    = batch["number_values"][batch["is_targets"]].flatten().float()
        mse  = F.mse_loss(pred, y).item()
        return mse, mse, mse, "mse"

    logits = yhat_dict["boolean"][batch["is_targets"]]
    y_raw  = batch["boolean_values"][batch["is_targets"]].flatten().float()

    mode = clf_mode.lower()
    if mode == "auto":
        if logits.dim() == 1 or (logits.dim() >= 2 and logits.shape[-1] == 1) or num_classes == 2:
            mode = "binary"
        else:
            mode = "multiclass"

    if mode == "binary":
        logits_1d = logits.flatten().float()
        y         = (y_raw > 0).float()
        bce       = F.binary_cross_entropy_with_logits(logits_1d, y).item()
        prob      = torch.sigmoid(logits_1d).detach().cpu().numpy()
        y_np      = y.detach().cpu().numpy().astype(int)
        if len(np.unique(y_np)) < 2:
            acc = accuracy_score(y_np, (prob >= 0.5).astype(int))
            return bce, acc, acc, "acc"
        auc = roc_auc_score(y_np, prob)
        return bce, auc, auc, "auc"

    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if logits.dim() != 2:
        logits = logits.view(logits.shape[0], -1)
    C     = max(logits.shape[-1], int(num_classes) if num_classes else 2)
    y_idx = torch.round(y_raw).long().clamp(min=0, max=C - 1)
    ce    = F.cross_entropy(logits.float(), y_idx).item()
    acc   = accuracy_score(y_idx.detach().cpu().numpy(), torch.argmax(logits.detach(), dim=-1).cpu().numpy())
    return ce, acc, acc, "acc"


@torch.no_grad()
def build_fixed_eval_cache(
    *,
    db_name: str,
    table_name: str,
    target_column: str,
    columns_to_drop: List[str],
    split: str,
    batch_size: int,
    seq_len: int,
    max_bfs_width: int,
    embedding_model: str,
    d_text: int,
    hop_cfg: HopConfig,
    num_workers: int,
    cache_batches: int,
    seed: int,
) -> List[dict]:
    assert split in {"val", "test"}

    eval_ds = RelationalDataset(
        tasks=[(db_name, table_name, target_column, split, columns_to_drop)],
        batch_size=batch_size, seq_len=seq_len, rank=0, world_size=1,
        max_bfs_width=max_bfs_width, embedding_model=embedding_model,
        d_text=d_text, seed=seed,
    )
    eval_ds.sampler.shuffle_py(0)

    loader = DataLoader(
        eval_ds, batch_size=None, num_workers=num_workers,
        persistent_workers=(num_workers > 0), pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    cache = []
    it    = iter(loader)

    for _ in range(cache_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        true_bs = int(batch.pop("true_batch_size"))
        batch["masks"][true_bs:, :]      = False
        batch["is_targets"][true_bs:, :] = False
        batch["is_padding"][true_bs:, :] = True

        apply_hop_control_from_p2f(batch, hop_cfg, sampler=eval_ds.sampler, dataset_idx=0, max_k=64)
        cache.append(batch)

    return cache


@torch.no_grad()
def eval_from_cache(
    net,
    cache: List[dict],
    *,
    device: torch.device,
    task_type: str,
    clf_mode: str,
    num_classes: int,
    ddp: bool,
) -> Tuple[float, float, float, str]:
    net.eval()
    losses           = []
    all_probs        = []
    all_ybin         = []
    all_acc          = []
    metric_name_seen = None

    for batch_cpu in cache:
        batch = _move_batch_to_device(batch_cpu, device)
        _, yhat_dict = net(batch)

        task_loss, score, human_metric, metric_name = _task_loss_and_metric(
            task_type=task_type, clf_mode=clf_mode,
            num_classes=num_classes, yhat_dict=yhat_dict, batch=batch,
        )
        metric_name_seen = metric_name
        losses.append(float(task_loss))

        if metric_name == "auc":
            logits = yhat_dict["boolean"][batch["is_targets"]].flatten().float()
            y      = (batch["boolean_values"][batch["is_targets"]].flatten().float() > 0).int()
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_ybin.append(y.detach().cpu().numpy())
        else:
            all_acc.append(float(human_metric))

    if not losses:
        bad_score = float("inf") if task_type == "reg" else float("-inf")
        return float("nan"), bad_score, float("nan"), "metric"

    avg_loss = sum(losses) / len(losses)

    if all_probs:
        probs = np.concatenate(all_probs)
        ybin  = np.concatenate(all_ybin)
        if len(np.unique(ybin)) < 2:
            acc = accuracy_score(ybin, (probs >= 0.5).astype(int))
            return avg_loss, acc, acc, "acc"
        auc = roc_auc_score(ybin, probs)
        return avg_loss, auc, auc, "auc"

    avg_metric = sum(all_acc) / max(len(all_acc), 1) if all_acc else float("nan")
    return avg_loss, avg_metric, avg_metric, (metric_name_seen or "acc")


@torch.no_grad()
def eval_from_cache_with_outputs(
    net,
    cache: List[dict],
    *,
    device: torch.device,
    task_type: str,
    clf_mode: str,
    num_classes: int,
) -> Tuple[float, float, str, Dict[str, np.ndarray]]:
    net.eval()
    losses      = []
    out_logits  = []
    out_probs   = []
    out_labels  = []
    out_preds   = []
    metric_name_final = None

    for batch_cpu in cache:
        batch = _move_batch_to_device(batch_cpu, device)
        _, yhat_dict = net(batch)

        task_loss, _, human_metric, metric_name = _task_loss_and_metric(
            task_type=task_type, clf_mode=clf_mode,
            num_classes=num_classes, yhat_dict=yhat_dict, batch=batch,
        )
        metric_name_final = metric_name
        losses.append(float(task_loss))

        if task_type == "reg":
            out_preds.append(yhat_dict["number"][batch["is_targets"]].flatten().float().detach().cpu().numpy())
            out_labels.append(batch["number_values"][batch["is_targets"]].flatten().float().detach().cpu().numpy())
        else:
            logits  = yhat_dict["boolean"][batch["is_targets"]]
            y_raw   = batch["boolean_values"][batch["is_targets"]].flatten().float()
            out_labels.append(y_raw.detach().cpu().numpy())
            out_logits.append(logits.detach().float().cpu().numpy())
            if logits.dim() == 1 or (logits.dim() >= 2 and logits.shape[-1] == 1) or clf_mode.lower() in {"binary"}:
                out_probs.append(torch.sigmoid(logits.flatten().float()).detach().cpu().numpy())

    avg_loss   = float("nan") if not losses else sum(losses) / len(losses)
    outputs: Dict[str, np.ndarray] = {}

    if task_type == "reg":
        preds  = np.concatenate(out_preds,  axis=0) if out_preds  else np.array([], dtype=np.float32)
        labels = np.concatenate(out_labels, axis=0) if out_labels else np.array([], dtype=np.float32)
        outputs["preds"]  = preds
        outputs["labels"] = labels
        metric_value      = float(mean_squared_error(labels, preds)) if preds.size and labels.size else float("nan")
        return avg_loss, metric_value, "mse", outputs

    logits_all  = None
    try:
        logits_all = np.concatenate(
            [x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(-1, 1) for x in out_logits], axis=0
        )
    except Exception:
        pass

    labels_raw = np.concatenate(out_labels, axis=0) if out_labels else np.array([], dtype=np.float32)
    outputs["labels_raw"] = labels_raw
    if logits_all is not None:
        outputs["logits"] = logits_all

    if out_probs:
        probs            = np.concatenate(out_probs, axis=0)
        outputs["probs"] = probs
        ybin             = (labels_raw > 0).astype(int)
        if np.unique(ybin).size < 2:
            metric_value = float(accuracy_score(ybin, (probs >= 0.5).astype(int)))
            return avg_loss, metric_value, "acc", outputs
        metric_value = float(roc_auc_score(ybin, probs))
        return avg_loss, metric_value, "auc", outputs

    if logits_all is None or logits_all.size == 0:
        return avg_loss, float("nan"), (metric_name_final or "acc"), outputs

    y_idx        = np.clip(np.round(labels_raw).astype(int), 0, max(int(logits_all.shape[1]) - 1, 0))
    metric_value = float(accuracy_score(y_idx, np.argmax(logits_all, axis=1)))
    return avg_loss, metric_value, "acc", outputs


def main(
    project="rt",
    eval_splits=("val",),
    max_eval_steps=40,
    train_steps_per_epoch=1000,
    load_ckpt_path=None,
    save_ckpt_dir="ckpts",
    compile_=False,
    seed=0,
    results_root="results",
    train_tasks=None,
    eval_tasks=None,
    batch_size=32,
    num_workers=8,
    max_bfs_width=256,
    lr=1e-4,
    wd=0.0,
    lr_schedule=False,
    max_grad_norm=1.0,
    embedding_model="all-MiniLM-L12-v2",
    d_text=384,
    seq_len=1024,
    num_blocks=12,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    max_hop=-1,
    fanout_per_hop="",
    clf_mode="auto",
    num_classes=2,
    max_steps=None,
    max_epochs=10,
    float_ckpt_path=None,
):
    assert train_tasks is not None and eval_tasks is not None, "Provide train_tasks and eval_tasks"

    seed_everything(seed)

    ddp    = "LOCAL_RANK" in os.environ
    device = torch.device("cuda")

    if ddp:
        os.environ["OMP_NUM_THREADS"] = str(num_workers)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank       = 0
        world_size = 1

    fanout_list = _parse_fanout(fanout_per_hop)
    hop_cfg     = _make_hop_cfg(int(max_hop), fanout_list)

    if rank == 0:
        print(f"[HOP CONTROL] max_hop={hop_cfg.max_hop} fanout={hop_cfg.fanout_per_hop} pad={hop_cfg.neighbor_pad_value}")

    use_wandb = (rank == 0) and (os.environ.get("WANDB_MODE", "").lower() != "disabled")
    if use_wandb:
        import wandb
        wandb.init(project=project, config={
            "batch_size": batch_size, "seq_len": seq_len,
            "max_hop": hop_cfg.max_hop, "fanout_per_hop": hop_cfg.fanout_per_hop,
            "lr": lr, "wd": wd, "num_blocks": num_blocks, "d_model": d_model,
            "clf_mode": clf_mode, "num_classes": num_classes, "seed": seed,
        })

    torch.set_num_threads(1)

    train_ds = RelationalDataset(
        tasks=[(db, tbl, tgt, "train", drop) for (db, tbl, tgt, drop) in train_tasks],
        batch_size=batch_size, seq_len=seq_len, rank=rank, world_size=world_size,
        max_bfs_width=max_bfs_width, embedding_model=embedding_model, d_text=d_text, seed=seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=None, num_workers=num_workers,
        persistent_workers=(num_workers > 0), pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    steps_per_epoch = len(train_loader)

    use_epoch_mode = (max_epochs is not None)
    if use_epoch_mode:
        max_steps = max_epochs * steps_per_epoch
        if rank == 0:
            print(f"Training Mode: EPOCHS | total_epochs={max_epochs} steps_per_epoch={steps_per_epoch} total_steps={max_steps}")
    else:
        if max_steps is None:
            raise ValueError("Provide max_epochs or max_steps")
        if rank == 0:
            print(f"Training Mode: STEPS | total_steps={max_steps}")

    (ev_db, ev_tbl, ev_tgt, ev_drop) = eval_tasks[0]
    task_type = "reg" if _is_regression_task(ev_tbl) else "clf"
    if rank == 0:
        print(f"[TASK] {ev_db}/{ev_tbl} type={task_type}")

    results_dir = None
    if rank == 0:
        results_dir = Path(results_root).expanduser() / str(ev_db) / str(ev_tbl) / str(max_hop)
        results_dir.mkdir(parents=True, exist_ok=True)

    val_cache  = None
    test_cache = None
    if rank == 0:
        if "val" in eval_splits:
            val_cache = build_fixed_eval_cache(
                db_name=ev_db, table_name=ev_tbl, target_column=ev_tgt, columns_to_drop=ev_drop,
                split="val", batch_size=batch_size, seq_len=seq_len, max_bfs_width=max_bfs_width,
                embedding_model=embedding_model, d_text=d_text, hop_cfg=hop_cfg,
                num_workers=num_workers, cache_batches=max_eval_steps, seed=0,
            )
        test_cache = build_fixed_eval_cache(
            db_name=ev_db, table_name=ev_tbl, target_column=ev_tgt, columns_to_drop=ev_drop,
            split="test", batch_size=batch_size, seq_len=seq_len, max_bfs_width=max_bfs_width,
            embedding_model=embedding_model, d_text=d_text, hop_cfg=hop_cfg,
            num_workers=num_workers, cache_batches=max_eval_steps, seed=0,
        )
        if task_type == "clf" and clf_mode.lower() == "auto":
            num_classes = int(_infer_num_classes_from_cache(val_cache or test_cache or []))
            print(f"[CLF] auto inferred num_classes={num_classes}")

    net = RelationalTransformer(
        num_blocks=num_blocks, d_model=d_model, d_text=d_text,
        num_heads=num_heads, d_ff=d_ff, float_ckpt_path=float_ckpt_path,
    )
    if load_ckpt_path is not None:
        net.load_state_dict(torch.load(Path(load_ckpt_path).expanduser(), map_location="cpu"))

    net = net.to(device).to(torch.bfloat16)
    if rank == 0:
        print(f"param_count={sum(p.numel() for p in net.parameters()):_}")

    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8, fused=True)

    lrs = None
    if lr_schedule:
        total_sched_steps = (max_epochs * min(steps_per_epoch, train_steps_per_epoch)) if use_epoch_mode else max_steps
        lrs = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=total_sched_steps, pct_start=0.2, anneal_strategy="linear",
        )

    if ddp:
        net = DDP(net)

    net = torch.compile(net, dynamic=False, disable=not compile_)

    def save_best_ckpt() -> Optional[str]:
        if rank != 0 or save_ckpt_dir is None:
            return None
        save_dir = Path(save_ckpt_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        path  = save_dir / f"{ev_db}_{ev_tbl}_best.pt"
        state = net.module.state_dict() if ddp else net.state_dict()
        torch.save(state, path)
        return str(path)

    best_val_metric = float("inf") if task_type == "reg" else float("-inf")
    best_ckpt_path  = None
    best_metric_name = None
    best_val_epoch  = None

    epoch_train_losses:     List[float] = []
    epoch_val_losses:       List[float] = []
    epoch_val_metrics:      List[float] = []
    epoch_val_metric_names: List[str]   = []
    metric_name_last = None

    pbar = tqdm(total=(max_epochs if use_epoch_mode else max_steps), disable=(rank != 0))

    global_step    = 0
    train_time_sec = 0.0

    if use_epoch_mode:
        for epoch in range(1, max_epochs + 1):
            if ddp:
                dist.barrier()

            t0 = time.time()
            train_loader.dataset.sampler.shuffle_py(epoch)
            it = iter(train_loader)
            net.train()

            loss_sum = 0.0
            loss_n   = 0
            steps_this_epoch = min(steps_per_epoch, train_steps_per_epoch)

            for _ in range(steps_this_epoch):
                try:
                    batch = next(it)
                except StopIteration:
                    break

                true_bs = int(batch.pop("true_batch_size"))
                batch   = _move_batch_to_device(batch, device)

                batch["masks"][true_bs:, :]      = False
                batch["is_targets"][true_bs:, :] = False
                batch["is_padding"][true_bs:, :] = True

                apply_hop_control_from_p2f(
                    batch, hop_cfg, sampler=train_loader.dataset.sampler, dataset_idx=0, max_k=64,
                )

                loss_pretrain, yhat_dict = net(batch)

                targets_mask = batch["is_targets"]
                has_targets  = targets_mask.any()

                if has_targets:
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

                grad_norm = get_total_norm([p.grad for p in net.parameters() if p.grad is not None])
                clip_grads_with_norm_(net.parameters(), max_norm=max_grad_norm, total_norm=grad_norm)
                opt.step()
                if lrs is not None:
                    lrs.step()

                if ddp:
                    dist.all_reduce(task_loss_bp, op=dist.ReduceOp.AVG)

                if rank == 0:
                    loss_sum += float(task_loss_bp.detach())
                    loss_n   += 1

                global_step += 1

            if ddp:
                dist.barrier()
            train_time_sec += time.time() - t0

            train_loss_epoch = (loss_sum / max(loss_n, 1)) if rank == 0 else None
            if rank == 0:
                epoch_train_losses.append(float(train_loss_epoch))

            val_loss_epoch = None
            val_human      = None
            metric_name    = "metric"

            if rank == 0 and val_cache is not None:
                val_loss_epoch, _, val_human, metric_name = eval_from_cache(
                    net.module if ddp else net, val_cache,
                    device=device, task_type=task_type, clf_mode=clf_mode,
                    num_classes=num_classes, ddp=ddp,
                )
                epoch_val_losses.append(float(val_loss_epoch))
                epoch_val_metrics.append(float(val_human))
                epoch_val_metric_names.append(metric_name)
                metric_name_last = metric_name

                is_better = (val_human < best_val_metric) if task_type == "reg" else (val_human > best_val_metric)
                if is_better:
                    best_val_metric  = float(val_human)
                    best_metric_name = metric_name
                    best_val_epoch   = epoch
                    best_ckpt_path   = save_best_ckpt()

            if rank == 0:
                pbar.set_postfix({
                    "train_loss":        f"{train_loss_epoch:.4f}" if train_loss_epoch is not None else "NA",
                    "val_loss":          f"{val_loss_epoch:.4f}"  if val_loss_epoch is not None  else "NA",
                    f"val_{metric_name}": f"{val_human:.4f}"      if val_human is not None       else "NA",
                })
                pbar.update(1)

            if use_wandb and rank == 0:
                import wandb
                wandb.log({
                    "epoch": epoch, "train_loss_epoch": train_loss_epoch,
                    "val_loss_epoch": val_loss_epoch, f"val_{metric_name}": val_human,
                    "best_val_metric": best_val_metric, "best_val_epoch": best_val_epoch,
                    "train_time_sec_cum": train_time_sec,
                }, step=global_step)

        pbar.close()

    else:
        while global_step < max_steps:
            t0 = time.time()
            train_loader.dataset.sampler.shuffle_py(global_step + 1)
            it = iter(train_loader)
            net.train()

            try:
                batch = next(it)
            except StopIteration:
                continue

            true_bs = int(batch.pop("true_batch_size"))
            batch   = _move_batch_to_device(batch, device)

            batch["masks"][true_bs:, :]      = False
            batch["is_targets"][true_bs:, :] = False
            batch["is_padding"][true_bs:, :] = True

            apply_hop_control_from_p2f(
                batch, hop_cfg, sampler=train_loader.dataset.sampler, dataset_idx=0, max_k=64,
            )

            loss_pretrain, yhat_dict = net(batch)

            targets_mask = batch["is_targets"]
            has_targets  = targets_mask.any()
            if has_targets:
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
            opt.step()

            global_step    += 1
            train_time_sec += time.time() - t0

            if rank == 0:
                pbar.update(1)

        pbar.close()

    test_time_sec      = 0.0
    val_outputs_paths  = {}
    test_outputs_paths = {}
    test_loss          = None
    test_human         = None
    test_metric_name   = metric_name_last or ("mse" if task_type == "reg" else "auc")

    if rank == 0:
        if best_ckpt_path is not None and test_cache is not None:
            state = torch.load(best_ckpt_path, map_location="cpu")
            model_for_eval = net.module if ddp else net
            if ddp:
                net.module.load_state_dict(state)
            else:
                net.load_state_dict(state)

            if val_cache is not None:
                val_loss_best, val_metric_best, val_metric_name_best, val_out = eval_from_cache_with_outputs(
                    model_for_eval, val_cache, device=device, task_type=task_type,
                    clf_mode=clf_mode, num_classes=num_classes,
                )
                if results_dir is not None:
                    if task_type == "reg":
                        np.save(results_dir / f"seed{seed}_val_preds.npy",   val_out.get("preds",  np.array([])))
                        np.save(results_dir / f"seed{seed}_val_labels.npy",  val_out.get("labels", np.array([])))
                        val_outputs_paths = {"preds": str(results_dir / f"seed{seed}_val_preds.npy"),
                                             "labels": str(results_dir / f"seed{seed}_val_labels.npy")}
                    else:
                        np.save(results_dir / f"seed{seed}_val_logits.npy",      val_out.get("logits",     np.array([])))
                        np.save(results_dir / f"seed{seed}_val_labels_raw.npy",  val_out.get("labels_raw", np.array([])))
                        val_outputs_paths = {"logits": str(results_dir / f"seed{seed}_val_logits.npy"),
                                             "labels_raw": str(results_dir / f"seed{seed}_val_labels_raw.npy")}
                        if "probs" in val_out:
                            np.save(results_dir / f"seed{seed}_val_probs.npy", val_out["probs"])
                            val_outputs_paths["probs"] = str(results_dir / f"seed{seed}_val_probs.npy")

            t0 = time.time()
            test_loss_best, test_metric_best, test_metric_name_best, test_out = eval_from_cache_with_outputs(
                model_for_eval, test_cache, device=device, task_type=task_type,
                clf_mode=clf_mode, num_classes=num_classes,
            )
            test_time_sec += time.time() - t0
            test_loss        = float(test_loss_best)
            test_human       = float(test_metric_best)
            test_metric_name = str(test_metric_name_best)

            if results_dir is not None:
                if task_type == "reg":
                    np.save(results_dir / f"seed{seed}_test_preds.npy",  test_out.get("preds",  np.array([])))
                    np.save(results_dir / f"seed{seed}_test_labels.npy", test_out.get("labels", np.array([])))
                    test_outputs_paths = {"preds":  str(results_dir / f"seed{seed}_test_preds.npy"),
                                          "labels": str(results_dir / f"seed{seed}_test_labels.npy")}
                else:
                    np.save(results_dir / f"seed{seed}_test_logits.npy",     test_out.get("logits",     np.array([])))
                    np.save(results_dir / f"seed{seed}_test_labels_raw.npy", test_out.get("labels_raw", np.array([])))
                    test_outputs_paths = {"logits":     str(results_dir / f"seed{seed}_test_logits.npy"),
                                          "labels_raw": str(results_dir / f"seed{seed}_test_labels_raw.npy")}
                    if "probs" in test_out:
                        np.save(results_dir / f"seed{seed}_test_probs.npy", test_out["probs"])
                        test_outputs_paths["probs"] = str(results_dir / f"seed{seed}_test_probs.npy")

        print("\n" + "=" * 90)
        print(f"BEST CHECKPOINT: task={ev_db}/{ev_tbl} type={task_type} seed={seed}")
        print(f"best_ckpt={best_ckpt_path}  best_val_epoch={best_val_epoch}")
        if task_type == "reg":
            print(f"best_val_mse={best_val_metric:.6f}")
            if test_human is not None:
                print(f"test_mse={test_human:.6f}  test_loss={test_loss:.6f}")
        else:
            print(f"best_val_{best_metric_name or test_metric_name}={best_val_metric:.6f}")
            if test_human is not None:
                print(f"test_{test_metric_name}={test_human:.6f}  test_loss={test_loss:.6f}")
        print(f"train_time_sec={train_time_sec:.3f}  test_time_sec={test_time_sec:.3f}")
        print("=" * 90 + "\n")

        if results_dir is not None:
            agg_path = results_dir / "results.json"
            agg      = json.loads(agg_path.read_text()) if agg_path.exists() else {}

            seed_payload = {
                "db": str(ev_db), "task": str(ev_tbl), "target_column": str(ev_tgt),
                "task_type": str(task_type), "seed": int(seed),
                "best_val_epoch": best_val_epoch,
                "best_val_metric_name": best_metric_name,
                "best_val_metric": _to_float(best_val_metric),
                "test_metric_name": test_metric_name,
                "test_metric": _to_float(test_human) if test_human is not None else None,
                "test_loss_avg_task_loss": _to_float(test_loss) if test_loss is not None else None,
                "train_time_sec": _to_float(train_time_sec),
                "test_time_sec":  _to_float(test_time_sec),
                "epoch_train_losses":      [float(x) for x in epoch_train_losses],
                "epoch_val_losses":        [float(x) for x in epoch_val_losses],
                "epoch_val_metrics":       [float(x) for x in epoch_val_metrics],
                "epoch_val_metric_names":  [str(x)   for x in epoch_val_metric_names],
                "best_ckpt_path":          str(best_ckpt_path) if best_ckpt_path is not None else None,
                "saved_val_outputs":       val_outputs_paths,
                "saved_test_outputs":      test_outputs_paths,
                "config": {
                    "batch_size": batch_size, "seq_len": seq_len, "max_bfs_width": max_bfs_width,
                    "embedding_model": embedding_model, "d_text": d_text,
                    "num_blocks": num_blocks, "d_model": d_model, "num_heads": num_heads, "d_ff": d_ff,
                    "lr": lr, "wd": wd, "lr_schedule": bool(lr_schedule), "max_grad_norm": max_grad_norm,
                    "max_hop": int(max_hop), "fanout_per_hop": hop_cfg.fanout_per_hop,
                    "max_eval_steps": int(max_eval_steps), "train_steps_per_epoch": int(train_steps_per_epoch),
                    "compile": bool(compile_),
                    "float_ckpt_path": str(float_ckpt_path) if float_ckpt_path is not None else None,
                    "load_ckpt_path":  str(load_ckpt_path)  if load_ckpt_path  is not None else None,
                },
            }

            agg.setdefault("seed_to_metric", {})[str(seed)] = {
                "metric_name": seed_payload["test_metric_name"],
                "metric":      seed_payload["test_metric"],
            }
            agg.setdefault("runs", {})[str(seed)] = seed_payload
            _safe_json_dump(agg, agg_path)
            _safe_json_dump({str(seed): seed_payload}, results_dir / f"seed{seed}.json")

    if rank == 0 and use_epoch_mode:
        import matplotlib.pyplot as plt
        xs = list(range(1, len(epoch_train_losses) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(xs, epoch_train_losses, label="Train Task Loss", marker="o")
        if epoch_val_losses:
            ys = list(range(1, len(epoch_val_losses) + 1))
            plt.plot(ys, epoch_val_losses, label="Val Task Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("RT Task Loss per Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve_epoch.png", dpi=150, bbox_inches="tight")
        plt.close()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import strictfire
    strictfire.StrictFire(main)
