import torch
import torch.nn.functional as F
from hdataset import Graph, Task
from hloaderwrapper import LoaderWrapperTask
from hmodel import GriffinMod
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from hFloatEmb import SimpleRepeater, getfloatdec
import numpy as np
import accelerate
import argparse
import os
import os.path as osp
import time
import json
from typing import Union, List
from metric import compute_metric, metric_higher_is_better
import yaml


import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))

def _infer_db_name(dataset_path: str) -> str:
    return osp.basename(dataset_path.rstrip("/"))

def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _plot_losses(csv_path: str, png_path: str):
    epochs, tr, va = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["valid_loss"]))
    if len(epochs) == 0:
        return
    plt.figure()
    plt.plot(epochs, tr, label="train")
    plt.plot(epochs, va, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=170)
    plt.close()

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def construct_dataset(graph, task, tasknames, split, args, floatembmodel):
    return LoaderWrapperTask(
        graph,
        batch_size=args.batchsize,
        subgraphargs={"floatemb": floatembmodel, "fanout": args.fanout, "hop": args.hop},
        shuffle=True if split == "train" else False,
        task=task,
        tasknames=tasknames,
        split=split,
        fewshotfanout=args.fewshotfanout,
    )


def compute_output(model, dec, data):
    label, y, mapping = data[-3:]
    data = data[:-3]
    if y is None:
        output = dec(model(*data)[mapping])
    else:
        output = model(*data)[mapping] @ y.T
    return output, label

def compute_loss(model, dec, data):
    label, y, mapping = data[-3:]
    data = data[:-3]
    if y is None:
        output = dec(model(*data)[mapping])
        loss = F.mse_loss(output.flatten(), label.flatten())
    else:
        output = model(*data)[mapping] @ y.T
        loss = F.cross_entropy(output, label)
    return loss


def eval_task_with_outputs(model, dec, dataset, accelerator, metric_name: str, save_outputs: bool = False):
    model.eval()
    dataset.rebuild_indice(accelerator)
    batchsize: int = dataset.batch_size

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda xlist: xlist[0],
        num_workers=8,
        persistent_workers=False,
    )
    loader = accelerator.prepare(loader)

    outputs_all = []
    labels_all = []

    with torch.no_grad():
        for data in loader:
            output, label = compute_output(model, dec, data)


            if output.shape[0] < batchsize:
                padnum = batchsize - output.shape[0]
                if torch.is_floating_point(label):
                    label = torch.concat(
                        (label, torch.empty_like(label[[0]].expand(padnum)).fill_(torch.nan)), dim=0
                    )
                else:
                    label = torch.concat(
                        (label, torch.empty_like(label[[0]].expand(padnum)).fill_(-1)), dim=0
                    )
                output = torch.concat(
                    (output, torch.empty_like(output[[0]].expand(padnum, -1)).fill_(torch.nan)), dim=0
                )

            output, label = output.unsqueeze(0), label.unsqueeze(0)
            output, label = accelerator.gather_for_metrics((output, label))
            output, label = output.flatten(0, 1), label.flatten(0, 1)

            if accelerator.is_main_process:
                if torch.is_floating_point(label):
                    mask = torch.isnan(label).logical_not_()
                else:
                    mask = label >= 0
                output, label = output[mask], label[mask]
                outputs_all.append(output.cpu())
                labels_all.append(label.cpu())

    if not accelerator.is_main_process:
        return None, None, None

    outputs_t = torch.concat(outputs_all, dim=0)
    labels_t = torch.concat(labels_all, dim=0)
    metric_val = compute_metric(outputs_t, labels_t, metric_name)

    if save_outputs:
        return metric_val, outputs_t.numpy(), labels_t.numpy()
    return metric_val, None, None


def eval_loss_epoch(model, dec, dataset, accelerator) -> Union[float, None]:
    model.eval()
    dataset.rebuild_indice(accelerator)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda xlist: xlist[0],
        num_workers=8,
        persistent_workers=False,
    )
    loader = accelerator.prepare(loader)

    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for data in loader:
            loss = compute_loss(model, dec, data)
            total_loss += float(loss.detach().cpu().item())
            total_n += 1

    t = torch.tensor([total_loss, total_n], device=accelerator.device)
    t = accelerator.gather_for_metrics(t).reshape(-1, 2)

    if accelerator.is_main_process:
        total_loss_all = t[:, 0].sum().item()
        total_n_all = int(t[:, 1].sum().item())
        return total_loss_all / max(total_n_all, 1)
    return None


def main(args):
    repo_root = osp.dirname(osp.abspath(__file__))
    db_name = _safe_name(_infer_db_name(args.dataset))


    graph = Graph(args.dataset)
    task = Task(args.dataset)

    tasknames = args.tasks
    if len(tasknames) == 1:
        if tasknames[0] == "ALLTASK":
            tasknames = [taskname for taskname in task.metatask]
        elif tasknames[0] == "RETTASK":
            tasknames = [t for t in task.metatask if task.metatask[t]["task_type"] == "retrieval"]
        elif tasknames[0] == "REGTASK":
            tasknames = [t for t in task.metatask if task.metatask[t]["task_type"] == "regression"]
        elif tasknames[0].startswith("EXCEPT__"):
            expect_taskname = tasknames[0][len("EXCEPT__"):]
            tasknames = [t for t in task.metatask if t != expect_taskname]
        elif tasknames[0] in ["commerce-1", "commerce-2", "others-1", "others-2"]:
            with open("task_names.yaml", "r") as f:
                tasks_dict = yaml.load(f, Loader=yaml.FullLoader)
            tasknames = tasks_dict[tasknames[0]]

    curve_task = tasknames[0]
    task_name = _safe_name(curve_task)
    metric_name = task.metatask[curve_task]["metric"]


    results_root = osp.join(repo_root, "results", db_name, task_name, str(args.hop))
    _ensure_dir(results_root)


    seeds = args.seed
    if isinstance(seeds, int):
        seeds = [seeds]

    summary_by_seed = {}

    for seed in seeds:
        seed = int(seed)
        seed_everything(seed)

        seed_dir = osp.join(results_root, f"seed_{seed}")
        _ensure_dir(seed_dir)

        loss_csv = osp.join(seed_dir, f"losses_{db_name}_{task_name}_seed{seed}.csv")
        loss_png = osp.join(seed_dir, f"losses_{db_name}_{task_name}_seed{seed}.png")

        with open(loss_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "valid_loss", "valid_metric"])

        ckpt_root = osp.join(repo_root, "checkpoints", _safe_name(args.logname), f"seed_{seed}")
        _ensure_dir(ckpt_root)
        best_ckpt_path = osp.join(ckpt_root, "best_checkpoint")

        tbconfig = ProjectConfiguration(project_dir=args.logdir, logging_dir=args.logdir)


        try:
            import tensorboard
            _log_with = "tensorboard"
        except ImportError:
            _log_with = None
        accelerator = Accelerator(log_with=_log_with, project_config=tbconfig)
        accelerator.init_trackers(f"{args.logname}_seed{seed}")
        tbtracker = accelerator.get_tracker("tensorboard") if _log_with else None

        model = GriffinMod(hiddim=args.hiddim, num_mp=args.num_mp, use_rev=args.use_rev, use_gate=args.use_gate)
        if args.loadpath is not None:
            accelerate.load_checkpoint_in_model(model, args.loadpath)
        dec = getfloatdec(args.hiddim)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        floatembmodel = SimpleRepeater(args.hiddim)
        train_dataset = construct_dataset(graph, task, tasknames, "train", args, floatembmodel)
        valid_dataset_dict = {t: construct_dataset(graph, task, [t], "valid", args, floatembmodel) for t in tasknames}
        test_dataset_dict  = {t: construct_dataset(graph, task, [t], "test",  args, floatembmodel) for t in tasknames}
        metric_dict = {t: task.metatask[t]["metric"] for t in tasknames}


        _curve_metric = metric_dict[curve_task]
        _higher_is_better = metric_higher_is_better(_curve_metric)
        best_val_metric = -float("inf") if _higher_is_better else float("inf")
        best_val_epoch = -1
        best_val_loss_at_best_metric = None


        epoch_train_losses: List[float] = []
        epoch_val_losses: List[float] = []
        epoch_val_metrics: List[float] = []
        epoch_val_metric_names: List[str] = []

        model, dec, optimizer = accelerator.prepare(model, dec, optimizer)


        train_time_start = time.time()
        step = 0

        for epoch in range(args.maxepoch):
            if accelerator.is_main_process:
                print(f"[seed {seed}] Epoch {epoch} starts")

            model.train()
            train_dataset.rebuild_indice(accelerator)
            loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=1,
                collate_fn=lambda xlist: xlist[0],
                num_workers=16,
                prefetch_factor=4,
                persistent_workers=False,
                pin_memory=True,
            )
            loader = accelerator.prepare(loader)

            epoch_train_loss_sum = 0.0
            epoch_train_steps = 0

            for data in loader:
                step += 1
                optimizer.zero_grad()
                loss = compute_loss(model, dec, data)

                epoch_train_loss_sum += float(loss.detach().cpu().item())
                epoch_train_steps += 1

                accelerator.backward(loss)


                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if step % 100 == 0:
                    if tbtracker is not None:
                        tbtracker.log({"training_loss": loss}, step=step)


            t = torch.tensor([epoch_train_loss_sum, epoch_train_steps], device=accelerator.device)
            t = accelerator.gather_for_metrics(t).reshape(-1, 2)
            train_loss_epoch = None
            if accelerator.is_main_process:
                train_loss_epoch = t[:, 0].sum().item() / max(int(t[:, 1].sum().item()), 1)

            accelerator.wait_for_everyone()


            valid_loss_epoch = eval_loss_epoch(model, dec, valid_dataset_dict[curve_task], accelerator)
            val_metric_epoch, _, _ = eval_task_with_outputs(
                model, dec, valid_dataset_dict[curve_task], accelerator, metric_dict[curve_task], save_outputs=False
            )

            if accelerator.is_main_process:
                epoch_train_losses.append(float(train_loss_epoch))
                epoch_val_losses.append(float(valid_loss_epoch))
                epoch_val_metrics.append(float(val_metric_epoch))
                epoch_val_metric_names.append(str(metric_dict[curve_task]))

                with open(loss_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss_epoch, valid_loss_epoch, _to_float(val_metric_epoch)])
                _plot_losses(loss_csv, loss_png)


                def _is_better(new_val, best_val):
                    if _higher_is_better:
                        return new_val > best_val
                    return new_val < best_val

                if val_metric_epoch is not None and _is_better(float(val_metric_epoch), best_val_metric):
                    best_val_metric = float(val_metric_epoch)
                    best_val_epoch = int(epoch)
                    best_val_loss_at_best_metric = _to_float(valid_loss_epoch)
                    print(f"[seed {seed}] [BEST] epoch={epoch} val_metric={best_val_metric:.6f} -> save {best_ckpt_path}")
                    accelerator.save_model(model, best_ckpt_path)


            if (epoch + 1) % args.eval_per_epoch == 0:
                if accelerator.is_main_process:
                    print(f"[seed {seed}] Metric logging @ epoch {epoch} ...")
                for tname in tasknames:
                    mname = metric_dict[tname]
                    metric_val, _, _ = eval_task_with_outputs(
                        model, dec, valid_dataset_dict[tname], accelerator, mname, save_outputs=False
                    )
                    if accelerator.is_main_process:
                        if tbtracker is not None:
                            tbtracker.log({f"valid_metric/{tname}/{mname}": metric_val}, step=step)
                        print(f"[seed {seed}] valid_metric/{tname}/{mname}: {metric_val}", flush=True)

        train_time_sec = time.time() - train_time_start


        if osp.exists(best_ckpt_path):
            if accelerator.is_main_process:
                print(f"[seed {seed}] Loading best checkpoint from {best_ckpt_path} (best_val_epoch={best_val_epoch}, best_val_metric={best_val_metric:.6f})")
            unwrap_model = accelerator.unwrap_model(model)
            accelerate.load_checkpoint_in_model(unwrap_model, best_ckpt_path)
            model = accelerator.prepare(unwrap_model)
        else:
            if accelerator.is_main_process:
                print(f"[seed {seed}] [WARN] No best checkpoint saved. Using last model state.")


        val_outputs_paths = []
        test_outputs_paths = []


        val_metric_best, val_logits, val_labels = eval_task_with_outputs(
            model, dec, valid_dataset_dict[curve_task], accelerator, metric_dict[curve_task], save_outputs=True
        )
        val_loss_best = eval_loss_epoch(model, dec, valid_dataset_dict[curve_task], accelerator)

        if accelerator.is_main_process:
            val_npz = osp.join(seed_dir, "val_outputs_best.npz")
            np.savez(val_npz, logits=val_logits, labels=val_labels)
            val_outputs_paths.append(val_npz)


        test_time_start = time.time()
        test_metric, test_logits, test_labels = eval_task_with_outputs(
            model, dec, test_dataset_dict[curve_task], accelerator, metric_dict[curve_task], save_outputs=True
        )
        test_time_sec = time.time() - test_time_start
        test_loss = eval_loss_epoch(model, dec, test_dataset_dict[curve_task], accelerator)

        if accelerator.is_main_process:
            test_npz = osp.join(seed_dir, "test_outputs_best.npz")
            np.savez(test_npz, logits=test_logits, labels=test_labels)
            test_outputs_paths.append(test_npz)


        if accelerator.is_main_process:
            seed_payload = {
                "db": str(db_name),
                "task": str(curve_task),
                "task_type": str(task.metatask[curve_task].get("task_type", "")),
                "seed": int(seed),

                "best_val_epoch": int(best_val_epoch),
                "best_val_metric_name": str(metric_dict[curve_task]),
                "best_val_metric": float(best_val_metric) if best_val_epoch >= 0 else None,
                "best_val_loss_at_best_metric_epoch": _to_float(best_val_loss_at_best_metric),

                "val_metric_name": str(metric_dict[curve_task]),
                "val_metric": _to_float(val_metric_best),
                "val_loss_avg_task_loss": _to_float(val_loss_best),

                "test_metric_name": str(metric_dict[curve_task]),
                "test_metric": _to_float(test_metric),
                "test_loss_avg_task_loss": _to_float(test_loss),

                "train_time_sec": float(train_time_sec),
                "test_time_sec": float(test_time_sec),

                "epoch_train_losses": [float(x) for x in epoch_train_losses],
                "epoch_val_losses": [float(x) for x in epoch_val_losses],
                "epoch_val_metrics": [float(x) for x in epoch_val_metrics],
                "epoch_val_metric_names": [str(x) for x in epoch_val_metric_names],

                "best_ckpt_path": str(best_ckpt_path) if osp.exists(best_ckpt_path) else None,
                "saved_val_outputs": val_outputs_paths,
                "saved_test_outputs": test_outputs_paths,

                "config": {
                    "batchsize": int(args.batchsize),
                    "eval_batchsize": int(args.eval_batchsize),
                    "lr": float(args.lr),
                    "wd": float(args.wd),
                    "maxepoch": int(args.maxepoch),
                    "eval_per_epoch": int(args.eval_per_epoch),
                    "hop": int(args.hop),
                    "fanout": list(args.fanout) if isinstance(args.fanout, list) else args.fanout,
                    "fewshotfanout": int(args.fewshotfanout),
                    "num_mp": int(args.num_mp),
                    "hiddim": int(args.hiddim),
                    "use_rev": bool(args.use_rev),
                    "use_gate": bool(args.use_gate),
                    "dataset": str(args.dataset),
                    "logdir": str(args.logdir),
                    "logname": str(args.logname),
                    "tasks": [str(x) for x in args.tasks],
                    "loadpath": str(args.loadpath) if args.loadpath is not None else None,
                },
            }

            with open(osp.join(seed_dir, "results.json"), "w") as f:
                json.dump(seed_payload, f, indent=2)

            summary_by_seed[str(seed)] = {
                "best_val_epoch": int(best_val_epoch),
                "best_val_metric": float(best_val_metric) if best_val_epoch >= 0 else None,
                "metric_name": str(metric_dict[curve_task]),
                "test_metric": _to_float(test_metric),
                "best_ckpt_path": str(best_ckpt_path) if osp.exists(best_ckpt_path) else None,
                "seed_dir": seed_dir,
            }

        accelerator.end_training()


    if len(summary_by_seed) > 0:
        summary_path = osp.join(results_root, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "db": db_name,
                    "task": curve_task,
                    "metric_name": metric_name,
                    "seeds": summary_by_seed,
                },
                f,
                indent=2,
            )
        print(f"[RESULTS] Saved summary: {summary_path}")
        print(f"[RESULTS] Root: {results_root}")


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if str(v).lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif str(v).lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("dataset", type=str)
    parser.add_argument("logdir", type=str)
    parser.add_argument("logname", type=str)

    parser.add_argument("--tasks", type=str, nargs="+", default=["ALLTASK"])
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--loadpath", type=str, default=None)


    parser.add_argument("--seed", type=int, nargs="+", default=[0])

    parser.add_argument("--batchsize", type=int, default=512)
    parser.add_argument("--eval_batchsize", type=int, default=None)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=4e-4)
    parser.add_argument("--maxepoch", type=int, default=10)
    parser.add_argument("--eval_per_epoch", type=int, default=3)

    parser.add_argument("--num_mp", type=int, default=4)
    parser.add_argument("--hiddim", type=int, default=256)


    parser.add_argument(
        "--fanout", type=int, nargs="*", default=None,
        help="Per-hop fanout list. hop=0 -> omit, hop=1 -> --fanout 10, hop=2 -> --fanout 10 30"
    )
    parser.add_argument("--fewshotfanout", type=int, default=0)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--use_rev", type=str2bool, default=True)
    parser.add_argument("--use_gate", type=str2bool, default=True)

    args = parser.parse_args()


    if args.hop == 0:
        args.fanout = []
    else:
        if args.fanout is None or len(args.fanout) == 0:
            args.fanout = [10] * args.hop
        elif len(args.fanout) == 1 and args.hop > 1:
            args.fanout = args.fanout * args.hop
        elif len(args.fanout) != args.hop:
            raise ValueError(f"--fanout must have length == --hop. Got hop={args.hop}, fanout={args.fanout}")

    args.eval_batchsize = args.batchsize if args.eval_batchsize is None else args.eval_batchsize

    main(args)