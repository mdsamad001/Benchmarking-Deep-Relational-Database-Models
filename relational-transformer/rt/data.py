import os

if os.environ.get("RUSTLER_HOOK", "0") == "1":
    import maturin_import_hook
    from maturin_import_hook.settings import MaturinSettings
    maturin_import_hook.install(settings=MaturinSettings(release=True, uv=True))

import json
from functools import cache
from typing import Any, Dict

import ml_dtypes  # noqa: F401
import numpy as np
import torch
from rustler import Sampler
from torch.utils.data import Dataset


def _preproc_root() -> str:
    home = os.environ.get("HOME", ".")
    return os.path.join(home, "scratch", "pre")


@cache
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


@cache
def _load_column_index(db_name: str) -> dict:
    path = os.path.join(_preproc_root(), db_name, "column_index.json")
    return _load_json(path)


def get_column_index(column_name: str, table_name: str, db_name: str) -> int:
    column_index = _load_column_index(db_name)
    target = f"{column_name} of {table_name}"
    if target not in column_index:
        raise ValueError(f'Column "{target}" not found in column_index.json for dataset {db_name}.')
    return int(column_index[target])


def _normalize_split(split: str) -> str:
    mapping = {"train": "Train", "val": "Val", "test": "Test"}
    return mapping.get(split, split)


def _select_table_info_key(table_info: Dict[str, Any], table_name: str, split: str) -> str:
    split_key = f"{table_name}:{split}"
    db_key    = f"{table_name}:Db"
    if split_key in table_info:
        return split_key
    if db_key in table_info:
        return db_key
    keys_preview = ", ".join(list(table_info.keys())[:30])
    raise KeyError(
        f"Missing table_info for table='{table_name}' split='{split}'. "
        f"Tried keys: '{split_key}' and '{db_key}'. "
        f"Available keys (first 30): {keys_preview}"
    )


def _bf16_from_packed_fp16(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.view(np.float16)).view(torch.bfloat16)


def _safe_reshape(t: torch.Tensor, *shape: int) -> torch.Tensor:
    return t.contiguous().reshape(*shape)


class RelationalDataset(Dataset):
    def __init__(self, tasks, batch_size, seq_len, rank, world_size,
                 max_bfs_width, embedding_model, d_text, seed):
        batch_size    = int(batch_size)
        seq_len       = int(seq_len)
        rank          = int(rank)
        world_size    = int(world_size)
        max_bfs_width = int(max_bfs_width)
        d_text        = int(d_text)
        seed          = int(seed)

        dataset_tuples         = []
        target_column_indices  = []
        drop_column_indices    = []

        for db_name, table_name, target_column, split, columns_to_drop in tasks:
            split_norm = _normalize_split(split)

            table_info_path = os.path.join(_preproc_root(), db_name, "table_info.json")
            table_info      = _load_json(table_info_path)
            table_info_key  = _select_table_info_key(table_info, table_name, split_norm)
            info            = table_info[table_info_key]

            node_idx_offset = int(info["node_idx_offset"])
            num_nodes       = int(info["num_nodes"])

            target_idx = get_column_index(target_column, table_name, db_name)
            target_column_indices.append(int(target_idx))

            drop_indices = [get_column_index(col, table_name, db_name) for col in columns_to_drop]
            drop_column_indices.append([int(x) for x in drop_indices])

            dataset_tuples.append((str(db_name), node_idx_offset, num_nodes))

        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=rank,
            world_size=world_size,
            max_bfs_width=max_bfs_width,
            embedding_model=str(embedding_model),
            d_text=d_text,
            seed=seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
        )

        self.seq_len = seq_len
        self.d_text  = d_text

    def __len__(self):
        return self.sampler.len_py()

    def __getitem__(self, batch_idx):
        tup = self.sampler.batch_py(batch_idx)
        out = dict(tup)

        for k, v in out.items():
            if k in ["number_values","datetime_values","text_values","col_name_values","boolean_values"]:
                out[k] = _bf16_from_packed_fp16(v)
            elif k == "true_batch_size":
                pass
            else:
                out[k] = torch.from_numpy(v)

        S, D = self.seq_len, self.d_text

        out["node_idxs"]        = _safe_reshape(out["node_idxs"],        -1, S)
        out["sem_types"]        = _safe_reshape(out["sem_types"],         -1, S)
        out["masks"]            = _safe_reshape(out["masks"],             -1, S)
        out["is_targets"]       = _safe_reshape(out["is_targets"],        -1, S)
        out["is_task_nodes"]    = _safe_reshape(out["is_task_nodes"],     -1, S)
        out["is_padding"]       = _safe_reshape(out["is_padding"],        -1, S)
        out["table_name_idxs"]  = _safe_reshape(out["table_name_idxs"],  -1, S)
        out["col_name_idxs"]    = _safe_reshape(out["col_name_idxs"],    -1, S)
        out["class_value_idxs"] = _safe_reshape(out["class_value_idxs"], -1, S)
        out["f2p_nbr_idxs"]     = _safe_reshape(out["f2p_nbr_idxs"],     -1, S, 5)
        out["number_values"]    = _safe_reshape(out["number_values"],     -1, S, 1)
        out["datetime_values"]  = _safe_reshape(out["datetime_values"],   -1, S, 1)
        out["boolean_values"]   = _safe_reshape(out["boolean_values"],    -1, S, 1).bfloat16()
        out["text_values"]      = _safe_reshape(out["text_values"],       -1, S, D)
        out["col_name_values"]  = _safe_reshape(out["col_name_values"],   -1, S, D)

        return out
