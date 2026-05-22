

import os
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightning as L
import lightning.pytorch.callbacks as L_callbacks
import numpy as np
import torch
import torch.utils.data as torch_data
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from simple_parsing import ArgumentParser

from relbench_dataset import RelBenchDBFormerDataset
from db_transformer.schema.schema import Schema

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

AggrType = Literal['sum', 'mean', 'min', 'max', 'cat']

RELBENCH_DATASETS = {
    'rel-amazon': ['user-churn', 'user-ltv'],
    'rel-stack':  ['user-engagement', 'post-votes'],
    'rel-trial':  ['study-outcome', 'study-adverse'],
    'rel-hm':     ['user-churn'],
    'rel-f1':     ['driver-position', 'driver-dnf'],
}


@dataclass
class ModelConfig:
    dim: int = 64
    aggr: AggrType = 'mean'
    batch_norm: bool = False
    layer_norm: bool = False


def _bfs_sample_nodes(
    data: HeteroData,
    seeds: Dict[str, torch.Tensor],
    num_hops: int,
    fanout: int,
) -> Dict[str, List[int]]:


    visited  = defaultdict(set)
    frontier = defaultdict(set)

    for nt, idx in seeds.items():
        for i in idx.cpu().tolist():
            visited[nt].add(i)
            frontier[nt].add(i)

    for _ in range(num_hops):
        next_frontier = defaultdict(set)

        for (src_type, _, dst_type), es in data.edge_items():
            if not hasattr(es, 'edge_index'):
                continue

            ei = es.edge_index.cpu()


            if frontier[src_type]:
                src_set = torch.tensor(sorted(frontier[src_type]), dtype=torch.long)
                mask    = torch.isin(ei[0], src_set)
                nbrs    = ei[1][mask].unique().tolist()
                if fanout > 0 and len(nbrs) > fanout:
                    nbrs = random.sample(nbrs, fanout)
                for n in nbrs:
                    if n not in visited[dst_type]:
                        visited[dst_type].add(n)
                        next_frontier[dst_type].add(n)


            if frontier[dst_type]:
                dst_set = torch.tensor(sorted(frontier[dst_type]), dtype=torch.long)
                mask    = torch.isin(ei[1], dst_set)
                nbrs    = ei[0][mask].unique().tolist()
                if fanout > 0 and len(nbrs) > fanout:
                    nbrs = random.sample(nbrs, fanout)
                for n in nbrs:
                    if n not in visited[src_type]:
                        visited[src_type].add(n)
                        next_frontier[src_type].add(n)

        frontier = next_frontier

    return {nt: sorted(nodes) for nt, nodes in visited.items()}


def _extract_subgraph(
    data: HeteroData,
    node_sets: Dict[str, List[int]],
    seed_type: str,
    seed_indices: List[int],
) -> HeteroData:


    sub     = HeteroData()
    id_maps = {}

    for nt, nodes in node_sets.items():
        if nt == seed_type:
            seed_set = set(seed_indices)
            others   = [n for n in nodes if n not in seed_set]
            ordered  = seed_indices + others
        else:
            ordered  = nodes


        idx    = torch.tensor(ordered, dtype=torch.long)
        id_map = {old: new for new, old in enumerate(ordered)}
        id_maps[nt] = id_map

        nd = data[nt]
        if hasattr(nd, 'tf'):

            sub[nt].tf = nd.tf[idx]
        elif hasattr(nd, 'x'):
            sub[nt].x  = nd.x.cpu()[idx]

        for attr in ('y', 'train_mask', 'val_mask', 'test_mask'):
            v = getattr(nd, attr, None)
            if v is not None:
                setattr(sub[nt], attr, v.cpu()[idx])

    sub[seed_type].batch_size = len(seed_indices)

    for (src_t, rel, dst_t), es in data.edge_items():
        if src_t not in id_maps or dst_t not in id_maps:
            continue
        if not hasattr(es, 'edge_index'):
            continue

        ei_cpu  = es.edge_index.cpu()
        src_map = id_maps[src_t]
        dst_map = id_maps[dst_t]
        new_src = torch.tensor([src_map.get(s, -1) for s in ei_cpu[0].tolist()])
        new_dst = torch.tensor([dst_map.get(d, -1) for d in ei_cpu[1].tolist()])
        keep    = (new_src >= 0) & (new_dst >= 0)
        if keep.any():
            sub[(src_t, rel, dst_t)].edge_index = torch.stack(
                [new_src[keep], new_dst[keep]]
            )

    return sub


class HeteroSubgraphLoader:


    def __init__(
        self,
        data: HeteroData,
        target_table: str,
        seed_indices: torch.Tensor,
        num_hops: int,
        fanout: int,
        batch_size: int,
        shuffle: bool = True,
        device=None,
    ):
        self.data         = data
        self.target_table = target_table
        self.seed_indices = seed_indices
        self.num_hops     = num_hops
        self.fanout       = fanout
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.device       = device

    def __len__(self) -> int:
        return math.ceil(len(self.seed_indices) / self.batch_size)

    def __iter__(self) -> Iterator[HeteroData]:
        idx = self.seed_indices.tolist()
        if self.shuffle:
            random.shuffle(idx)

        for start in range(0, len(idx), self.batch_size):
            batch_seeds = idx[start : start + self.batch_size]

            node_sets = _bfs_sample_nodes(
                self.data,
                seeds={self.target_table: torch.tensor(batch_seeds, dtype=torch.long)},
                num_hops=self.num_hops,
                fanout=self.fanout,
            )


            for nt in self.data.node_types:
                if nt not in node_sets:
                    node_sets[nt] = []

            sub = _extract_subgraph(
                self.data, node_sets,
                seed_type=self.target_table,
                seed_indices=batch_seeds,
            )

            if self.device is not None:
                sub = sub.to(self.device)
            yield sub


class TheLightningModel(L.LightningModule):


    def __init__(
        self,
        model: torch.nn.Module,
        target_table: str,
        task_type: str,
        lr: float,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model        = model
        self.lr           = lr
        self.target_table = target_table

        if "classification" in task_type.lower():
            self.is_classification = True
            if class_weights is not None:
                self.register_buffer("class_weights", class_weights)
                self.loss_module = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.loss_module = torch.nn.CrossEntropyLoss()
        else:
            self.is_classification = False
            self.loss_module = torch.nn.MSELoss()

    def _step(self, batch: HeteroData):
        tf_dict   = {nt: batch[nt].tf        for nt in batch.node_types
                     if hasattr(batch[nt], 'tf')}
        edge_dict = {et: batch[et].edge_index for et in batch.edge_types
                     if hasattr(batch[et], 'edge_index')}

        out    = self.model(tf_dict, edge_dict)
        n_seed = batch[self.target_table].batch_size
        logits = out[:n_seed]
        y      = batch[self.target_table].y[:n_seed]

        if self.is_classification:
            valid   = y >= 0
            logits  = logits[valid]
            targets = y[valid].long()
            loss    = self.loss_module(logits, targets)
            metric  = (logits.argmax(-1) == targets).float().mean()
        else:
            preds   = logits.squeeze(-1)
            targets = y.float()


            valid   = ~torch.isnan(targets)
            if valid.sum() == 0:

                dummy = preds.mean() * 0.0
                return dummy, dummy

            preds   = preds[valid]
            targets = targets[valid]
            loss    = self.loss_module(preds, targets)

            metric  = loss.detach()

        return loss, metric

    def configure_optimizers(self):
        opt   = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=10)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_metric"}}

    def training_step(self, batch, _):
        loss, metric = self._step(batch)
        bs = batch[self.target_table].batch_size
        self.log("train_loss",   loss,   batch_size=bs, prog_bar=True)
        self.log("train_metric", metric, batch_size=bs, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, metric = self._step(batch)
        bs = batch[self.target_table].batch_size
        self.log("val_loss",   loss,   batch_size=bs, prog_bar=True)
        self.log("val_metric", metric, batch_size=bs, prog_bar=True)

    def test_step(self, batch, _):
        loss, metric = self._step(batch)
        bs = batch[self.target_table].batch_size
        self.log("test_loss",   loss,   batch_size=bs, prog_bar=True)
        self.log("test_metric", metric, batch_size=bs, prog_bar=True)


def create_data_from_relbench(
    dataset_name, task_name, device=None,
    use_text_embeddings=False, drop_timestamps=True,
):
    rds = RelBenchDBFormerDataset(
        dataset_name=dataset_name, task_name=task_name,
        data_dir="./relbench_data",
        use_text_embeddings=use_text_embeddings,
        drop_timestamps=drop_timestamps,
    )
    data, col_stats_dict, table_dfs = rds.build_hetero_data(device=device)

    colnames = {
        tbl: dict(data[tbl].tf.col_names_dict)
        for tbl in rds.schema.keys()
        if tbl in data and hasattr(data[tbl], 'tf')
    }

    tn = data[rds.target_table]
    if not all(hasattr(tn, m) for m in ("train_mask", "val_mask", "test_mask")):
        print("[WARN] RelBench masks missing — using RandomNodeSplit")
        n    = tn.tf.num_rows
        data = T.RandomNodeSplit("train_rest",
                                 num_val=int(0.2*n), num_test=int(0.1*n))(data)
    else:
        print("Using RelBench train/val/test splits")

    return data, rds.schema, col_stats_dict, colnames, rds


def create_model(
    data, col_stats_dict, rds,
    model_config: ModelConfig,
    num_hops: int,
    device=None,
    use_griffin=False,
    griffin_pretrained_float_path=None,
):
    from db_transformer.nn.models.blueprint import BlueprintModel

    task_str = str(rds.task.task_type)
    is_cls   = "classification" in task_str.lower()
    y        = data[rds.target_table].y

    out_dim  = (int(y[y >= 0].max().item()) + 1) if is_cls else 1
    print(f"[MODEL] {'Classification' if is_cls else 'Regression'}, out_dim={out_dim}")

    col_names_dict_per_table = {
        nt: data[nt].tf.col_names_dict
        for nt in data.node_types if hasattr(data[nt], 'tf')
    }
    edge_types = [et for et in data.edge_types if len(et) == 3]

    return BlueprintModel(
        target=(rds.target_table, rds.target_column),
        embed_dim=model_config.dim,
        col_stats_per_table=col_stats_dict,
        col_names_dict_per_table=col_names_dict_per_table,
        edge_types=edge_types,
        num_gnn_layers=num_hops,
        positional_encoding=False,
        decoder=torch.nn.Sequential(
            torch.nn.LayerNorm(model_config.dim),
            torch.nn.Linear(model_config.dim, out_dim),
        ),
        decoder_aggregation=lambda x: x.mean(dim=1) if x.ndim == 3 else x,
        output_activation=None,
        use_griffin_embedder=use_griffin,
        griffin_device=device,
        griffin_pretrained_float_path=griffin_pretrained_float_path,
        griffin_table_dataframes=rds.table_dfs,
    )


class MetricPlotCallback(L.Callback):
    def __init__(self, save_path="plots/curves.png"):
        super().__init__()
        self.save_path = save_path
        self.epochs, self.losses, self.t_met, self.v_met = [], [], [], []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.epochs.append(trainer.current_epoch)
        self.losses.append(m.get("train_loss",   torch.tensor(float('nan'))).item())
        self.t_met.append( m.get("train_metric", torch.tensor(float('nan'))).item())
        self.v_met.append( m.get("val_metric",   torch.tensor(float('nan'))).item())
        self._plot()

    def on_fit_end(self, *_): self._plot()

    def _plot(self):
        if not self.epochs: return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.epochs, self.losses, label="train_loss"); ax.set_xlabel("Epoch")
        ax2 = ax.twinx()
        ax2.plot(self.epochs, self.t_met, "--", label="train_metric")
        ax2.plot(self.epochs, self.v_met, ":",  label="val_metric")
        h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc="upper right")
        fig.tight_layout()
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        fig.savefig(self.save_path); plt.close(fig)


def main(
    dataset_name: str,
    task_name: str,
    model_config: Optional[ModelConfig] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    num_hops: int = 2,
    fanout: int = 10,
    cuda: bool = False,
    use_text_embeddings: bool = False,
    drop_timestamps: bool = True,
    use_griffin: bool = False,
    griffin_pretrained_float_path: Optional[str] = None,
):


    if model_config is None:
        model_config = ModelConfig()

    device = 'cuda' if (cuda and torch.cuda.is_available()) else 'cpu'

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name} / {task_name}")
    print(f"Hops    : {num_hops}  (GNN depth = {num_hops} layers)")
    print(f"Fan-out : {fanout}  neighbours per hop  (-1 = all)")
    print(f"Batch   : {batch_size} seed nodes per mini-batch")
    print(f"Device  : {device}")
    print(f"{'='*60}\n")


    data, schema, col_stats, colnames, rds = create_data_from_relbench(
        dataset_name, task_name, device,
        use_text_embeddings=use_text_embeddings,
        drop_timestamps=drop_timestamps,
    )
    tt       = rds.target_table
    task_str = str(rds.task.task_type)
    is_cls   = "classification" in task_str.lower()

    train_idx = data[tt].train_mask.nonzero(as_tuple=True)[0]
    val_idx   = data[tt].val_mask.nonzero(as_tuple=True)[0]
    test_idx  = data[tt].test_mask.nonzero(as_tuple=True)[0]
    print(f"Split — train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")


    class_weights = None
    if is_cls:
        y_train = data[tt].y[data[tt].train_mask].long()
        nc      = int(y_train.max().item()) + 1
        counts  = torch.bincount(y_train, minlength=nc).float()
        class_weights = float(y_train.numel()) / (nc * counts)
        print(f"Class counts:  {counts.tolist()}")
        print(f"Class weights: {class_weights.tolist()}")
    else:

        y_all   = data[tt].y
        y_valid = y_all[~torch.isnan(y_all)]
        print(f"Regression label stats (non-NaN): "
              f"n={len(y_valid)}, "
              f"min={y_valid.min().item():.3f}, "
              f"max={y_valid.max().item():.3f}, "
              f"mean={y_valid.mean().item():.3f}")


    print(f"\nBuilding HeteroSubgraphLoaders "
          f"(num_hops={num_hops}, fanout={fanout}, batch_size={batch_size})...")

    train_loader = HeteroSubgraphLoader(
        data, tt, train_idx, num_hops, fanout, batch_size,
        shuffle=True,  device=device)
    val_loader   = HeteroSubgraphLoader(
        data, tt, val_idx,   num_hops, fanout, batch_size,
        shuffle=False, device=device)
    test_loader  = HeteroSubgraphLoader(
        data, tt, test_idx,  num_hops, fanout, batch_size,
        shuffle=False, device=device)

    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val   batches/epoch: {len(val_loader)}")


    model = create_model(
        data, col_stats, rds, model_config,
        num_hops=num_hops, device=device,
        use_griffin=use_griffin,
        griffin_pretrained_float_path=griffin_pretrained_float_path,
    )

    lit = TheLightningModel(
        model, target_table=tt, task_type=task_str,
        lr=learning_rate, class_weights=class_weights,
    )


    plot_cb = MetricPlotCallback(f"plots/{dataset_name}-{task_name}-curves.png")


    monitor_mode = 'max' if is_cls else 'min'

    trainer = L.Trainer(
        accelerator='gpu' if (cuda and torch.cuda.is_available()) else 'cpu',
        devices=1,
        deterministic=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            L_callbacks.Timer(),
            L_callbacks.ModelCheckpoint(
                './torch-models/', save_top_k=1,
                filename=f'{dataset_name}-{task_name}-{{epoch}}-{{val_metric:.3f}}',
                mode=monitor_mode, monitor='val_metric',
            ),
            L_callbacks.EarlyStopping(
                monitor='val_metric', mode=monitor_mode, patience=20, verbose=True,
            ),
            plot_cb,
        ],
        max_epochs=epochs,
    )

    trainer.fit(lit, train_loader, val_loader)
    print("=== Training done ===")
    trainer.test(lit, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dataset', choices=list(RELBENCH_DATASETS.keys()))
    parser.add_argument('task',    type=str)
    parser.add_argument("--epochs",      "-e", type=int,   default=100)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--batch-size",  "-b", type=int,   default=64)
    parser.add_argument("--num-hops",    "-H", type=int,   default=2,
        help="Hop radius from each seed. Sets GNN depth = num_hops.")
    parser.add_argument("--fanout",      "-f", type=int,   default=10,
        help="Max neighbours sampled per node per hop. -1 = all neighbours.")
    parser.add_argument("--cuda",              action="store_true")
    parser.add_argument("--use-text-embeddings", action="store_true")
    parser.add_argument("--keep-timestamps",     action="store_true")
    parser.add_arguments(ModelConfig, dest="model_config")
    parser.add_argument("--use-griffin",           action="store_true")
    parser.add_argument("--griffin-float-weights", type=str, default=None)

    args = parser.parse_args()
    if args.task not in RELBENCH_DATASETS[args.dataset]:
        print(f"Error: '{args.task}' not in {RELBENCH_DATASETS[args.dataset]}")
        exit(1)

    main(
        dataset_name=args.dataset, task_name=args.task,
        model_config=args.model_config,
        epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size,
        num_hops=args.num_hops,
        fanout=args.fanout,
        cuda=args.cuda,
        use_text_embeddings=args.use_text_embeddings,
        drop_timestamps=not args.keep_timestamps,
        use_griffin=args.use_griffin,
        griffin_pretrained_float_path=args.griffin_float_weights,
    )