

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def extract_foreign_keys(dataset) -> List[Tuple[str, str, str, str]]:

    db = dataset.get_db() if hasattr(dataset, 'get_db') else dataset.db
    foreign_keys = []

    for table_name, table_obj in db.table_dict.items():
        if hasattr(table_obj, 'fkey_col_to_pkey_table'):
            fkey_dict = table_obj.fkey_col_to_pkey_table
            for fk_col, target_table in fkey_dict.items():
                target_obj = db.table_dict[target_table]
                pk_col = target_obj.pkey_col if hasattr(target_obj, 'pkey_col') and target_obj.pkey_col else target_obj.df.columns[0]
                foreign_keys.append((table_name, fk_col, target_table, pk_col))

    return foreign_keys


def build_graph(dataset, foreign_keys, max_rows=None):

    db = dataset.get_db() if hasattr(dataset, 'get_db') else dataset.db
    data = HeteroData()
    table_dfs = {}


    for table_name, table_obj in db.table_dict.items():
        df = table_obj.df.copy()
        if max_rows and table_name in max_rows:
            df = df.head(max_rows[table_name])

        table_dfs[table_name] = df


        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if 'id' not in c.lower()]

        if len(num_cols) > 0:
            features = df[num_cols].fillna(0).astype(np.float32).values
        else:
            features = np.ones((len(df), 1), dtype=np.float32)

        data[table_name].x = torch.from_numpy(features)
        data[table_name].num_nodes = len(df)


    for src_table, src_col, dst_table, dst_col in foreign_keys:
        src_df = table_dfs[src_table]
        dst_df = table_dfs[dst_table]

        if src_col not in src_df.columns or dst_col not in dst_df.columns:
            continue


        dst_map = {val: idx for idx, val in enumerate(dst_df[dst_col].values)}

        src_idx, dst_idx = [], []
        for i, val in enumerate(src_df[src_col].values):
            if pd.notna(val) and val in dst_map:
                src_idx.append(i)
                dst_idx.append(dst_map[val])

        if len(src_idx) > 0:
            edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)


            data[src_table, 'to', dst_table].edge_index = edge_index


            edge_index_rev = torch.tensor([dst_idx, src_idx], dtype=torch.long)
            data[dst_table, 'rev_to', src_table].edge_index = edge_index_rev

    return data, table_dfs


def get_labels(task, table_dfs):

    train_table = task.get_table('train')
    val_table = task.get_table('val')
    test_table = task.get_table('test')


    train_df = train_table.df if hasattr(train_table, 'df') else train_table
    val_df = val_table.df if hasattr(val_table, 'df') else val_table
    test_df = test_table.df if hasattr(test_table, 'df') else test_table

    print(f"  Task columns: {list(train_df.columns)}")
    print(f"  Target column: {task.target_col}")


    entity_col = None
    for col in train_df.columns:
        if 'timestamp' not in col.lower() and 'date' not in col.lower() and col != task.target_col:
            entity_col = col
            break

    if entity_col is None:
        entity_col = train_df.columns[0]

    print(f"  Entity column: {entity_col}")


    target_df = table_dfs[task.entity_table]


    id_col = None
    for col in target_df.columns:
        if 'id' in col.lower():
            id_col = col
            break
    if id_col is None:
        id_col = target_df.columns[0]

    print(f"  Graph ID column: {id_col}")

    id_to_idx = {val: idx for idx, val in enumerate(target_df[id_col].values)}
    print(f"  Graph has {len(id_to_idx)} {task.entity_table} nodes")

    def build_split(split_df, split_name):

        entity_ids = split_df[entity_col].values


        valid_mask = np.array([id in id_to_idx for id in entity_ids])
        valid_ids = entity_ids[valid_mask]

        print(f"  {split_name}: {len(valid_ids)}/{len(entity_ids)} entities found in graph")

        if len(valid_ids) == 0:
            print(f"  ⚠ WARNING: No matching entities! Check if entity IDs match.")
            print(f"    Task entity IDs (first 5): {entity_ids[:5]}")
            print(f"    Graph node IDs (first 5): {list(id_to_idx.keys())[:5]}")


        indices = torch.tensor([id_to_idx[id] for id in valid_ids], dtype=torch.long)


        if task.target_col in split_df.columns:
            labels_raw = split_df[task.target_col].values[valid_mask]
        elif 'label' in split_df.columns:
            labels_raw = split_df['label'].values[valid_mask]
        elif len(split_df.columns) >= 2:

            label_cols = [c for c in split_df.columns if 'timestamp' not in c.lower() and 'date' not in c.lower() and c != entity_col]
            if label_cols:
                labels_raw = split_df[label_cols[-1]].values[valid_mask]
            else:
                labels_raw = split_df.iloc[:, -1].values[valid_mask]
        else:
            raise KeyError(f"Cannot find label column. Available: {split_df.columns.tolist()}")

        labels = torch.tensor(labels_raw, dtype=torch.long)


        assert len(indices) == len(labels), f"Misalignment! indices: {len(indices)}, labels: {len(labels)}"

        return labels, indices

    train_labels, train_idx = build_split(train_df, "Train")
    val_labels, val_idx = build_split(val_df, "Val")
    test_labels, test_idx = build_split(test_df, "Test")

    return train_labels, val_labels, test_labels, train_idx, val_idx, test_idx


class TableEncoder(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.encoder(x)


class HeteroGNN(nn.Module):

    def __init__(self, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        return x


class DBFormer(nn.Module):

    def __init__(self, node_types, edge_types, metadata, hidden_channels=128, num_classes=2):
        super().__init__()


        self.encoders = nn.ModuleDict()
        for node_type in node_types:
            in_channels = metadata[node_type]['num_features']
            self.encoders[node_type] = TableEncoder(in_channels, hidden_channels)


        if len(edge_types) > 0:
            self.gnn = HeteroGNN(hidden_channels)
            self.gnn = to_hetero(self.gnn, metadata=(node_types, edge_types), aggr='mean')
        else:
            self.gnn = None


        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, num_classes)
        )

        self.target_node_type = None

    def forward(self, x_dict, edge_index_dict, target_node_type, batch_idx=None):

        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.encoders[node_type](x)


        if self.gnn is not None and len(edge_index_dict) > 0:
            h_dict = self.gnn(h_dict, edge_index_dict)


        h = h_dict[target_node_type]


        if batch_idx is not None:
            h = h[batch_idx]


        return self.classifier(h)


def train_epoch(model, data, target_type, train_idx, train_labels, optimizer, device):

    model.train()
    optimizer.zero_grad()

    try:
        out = model(data.x_dict, data.edge_index_dict, target_type, train_idx)


        if out.shape[0] != train_labels.shape[0]:
            raise ValueError(f"Dimension mismatch! out: {out.shape}, labels: {train_labels.shape}")

        loss = F.cross_entropy(out, train_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = (pred == train_labels).float().mean()

        return loss.item(), acc.item()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ CUDA OOM during training!")
            torch.cuda.empty_cache()
            raise
        elif "invalid configuration" in str(e):
            print(f"❌ CUDA kernel error! This might be a dimension issue")
            print(f"Shapes: out={out.shape if 'out' in locals() else 'N/A'}")
            raise
        else:
            raise


@torch.no_grad()
def evaluate(model, data, target_type, idx, labels, device):

    model.eval()
    out = model(data.x_dict, data.edge_index_dict, target_type, idx)
    loss = F.cross_entropy(out, labels)
    pred = out.argmax(dim=1)
    acc = (pred == labels).float().mean()
    return loss.item(), acc.item()


def main():
    print("="*80)
    print("DBFormer + RelBench - COMPLETE TRAINING")
    print("="*80)


    dataset_name = 'rel-f1'
    task_name = 'driver-top3'


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")


    print("\n[1/5] Loading RelBench data...")
    dataset = get_dataset(dataset_name)
    task = get_task(dataset_name, task_name)


    print("[2/5] Extracting foreign keys...")
    fkeys = extract_foreign_keys(dataset)
    print(f"✓ Found {len(fkeys)} foreign keys")


    print("[3/5] Building graph...")
    data, table_dfs = build_graph(dataset, fkeys, max_rows=None)
    print(f"✓ {len(data.node_types)} node types, {len(data.edge_types)} edge types")


    print("[4/5] Preparing labels...")
    train_labels, val_labels, test_labels, train_idx, val_idx, test_idx = get_labels(task, table_dfs)


    train_labels = train_labels.cpu().long()
    val_labels   = val_labels.cpu().long()
    test_labels  = test_labels.cpu().long()


    all_labels = torch.cat([train_labels, val_labels, test_labels])
    unique_labels = torch.unique(all_labels, sorted=True)

    print("All label classes (raw):", unique_labels)


    label2idx = {int(v.item()): i for i, v in enumerate(unique_labels)}

    def remap(labels: torch.Tensor) -> torch.Tensor:
        mapped = torch.empty_like(labels, dtype=torch.long)
        for raw, new in label2idx.items():
            mapped[labels == raw] = new
        return mapped


    train_labels = remap(train_labels)
    val_labels   = remap(val_labels)
    test_labels  = remap(test_labels)


    num_classes = len(unique_labels)
    print(f"✓ {num_classes} classes (reindexed to 0..{num_classes-1})")


    print(f"✓ {len(train_labels)} train, {len(val_labels)} val, {len(test_labels)} test")
    print(f"✓ {num_classes} classes")


    print("[5/5] Creating model...")
    metadata = {nt: {'num_features': data[nt].x.shape[1]} for nt in data.node_types}


    print(f"✓ Metadata: {[(k, v['num_features']) for k, v in metadata.items()]}")

    try:
        model = DBFormer(
            node_types=data.node_types,
            edge_types=data.edge_types,
            metadata=metadata,
            hidden_channels=128,
            num_classes=num_classes
        ).to(device)
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ CUDA OOM! Try smaller hidden_channels or CPU")
            torch.cuda.empty_cache()
            raise
        else:
            raise


    try:
        data = data.to(device)
        train_labels = train_labels.to(device)
        val_labels = val_labels.to(device)
        test_labels = test_labels.to(device)
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ CUDA OOM moving data! Graph too large for GPU")
            print(f"Suggestion: Use CPU or reduce max_rows")
            torch.cuda.empty_cache()
            raise
        else:
            raise

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    best_val_acc = 0
    for epoch in range(1, 51):
        train_loss, train_acc = train_epoch(
            model, data, task.entity_table, train_idx, train_labels, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, data, task.entity_table, val_idx, val_labels, device
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")


    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_acc = evaluate(model, data, task.entity_table, test_idx, test_labels, device)

    print(f"Best Val Acc:  {best_val_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Acc:      {test_acc:.4f}")
    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()