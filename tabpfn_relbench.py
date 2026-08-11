#!/usr/bin/env python3
import os
import gc
import json
import time
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names
from relbench.base import TaskType, EntityTask, Table, Database

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion


DEFAULT_FANOUT_HOP1 = 10
DEFAULT_FANOUT_HOP2 = 30

NUM_AGGS = ("sum", "mean")
CAT_AGGS = () 


def ensure_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")


def get_table_obj(task: EntityTask, split: str) -> Table:
    tbl = task.get_table(split, mask_input_cols=False)
    if isinstance(tbl, Table):
        return tbl
    if hasattr(tbl, "df") and hasattr(tbl, "time_col"):
        return tbl
    raise RuntimeError(
        f"task.get_table('{split}', mask_input_cols=False) did not return a relbench.base.Table. "
        f"Got type={type(tbl)}. Refusing to guess."
    )


def print_first5_targets(task: EntityTask, split: str):
    tbl = get_table_obj(task, split)
    df = tbl.df
    if task.target_col in df.columns:
        y = df[task.target_col]
        print(f"[TARGET HEAD] split={split} col={task.target_col} dtype={y.dtype}")
        print(y.head(5).to_string(index=False))
    else:
        print(f"[TARGET HEAD] split={split} has NO target column '{task.target_col}'")


def infer_mode_from_task(task: EntityTask) -> str:
    """
    Robust parsing of RelBench TaskType.
    Avoid brittle enum equality comparisons (can vary across versions).
    """
    tt = task.task_type
    name = getattr(tt, "name", str(tt)).lower()

    if "regress" in name:
        return "regression"
    if "binary" in name:
        return "binary"
    if "multi" in name:
        return "multiclass"
    raise RuntimeError(f"Unsupported task_type={tt} (parsed name={name})")


def infer_mode_with_target_override(task: EntityTask, y: pd.Series) -> str:
    """
    Trust RelBench task type first.
    If it says binary but target looks continuous (many unique numeric values),
    force regression to prevent bogus AUC printing.
    """
    mode = infer_mode_from_task(task)

    y_s = pd.Series(y)
    y_num = pd.to_numeric(y_s, errors="coerce")
    nunq = int(y_num.nunique(dropna=True))

    print("\n[MODE DEBUG]")
    print("task.task_type:", task.task_type)
    print("target dtype:", y_s.dtype)
    print("target nunique (numeric):", nunq)

    if mode == "binary" and nunq >= 20:
        print("[OVERRIDE] Target looks continuous → forcing REGRESSION mode.")
        return "regression"

    return mode


def drop_pkey_fkey_time(df: pd.DataFrame, table_obj) -> pd.DataFrame:
    out = df.copy()
    if getattr(table_obj, "pkey_col", None) is not None and table_obj.pkey_col in out.columns:
        out = out.drop(columns=[table_obj.pkey_col])
    f2p = getattr(table_obj, "fkey_col_to_pkey_table", {}) or {}
    fk_cols = [c for c in f2p.keys() if c in out.columns]
    if fk_cols:
        out = out.drop(columns=fk_cols)
    if getattr(table_obj, "time_col", None) is not None and table_obj.time_col in out.columns:
        out = out.drop(columns=[table_obj.time_col])
    return out


def _rng_for(seed: int, split: str) -> np.random.RandomState:
    h = abs(hash((seed, split))) % (2**31 - 1)
    return np.random.RandomState(h)


def sanitize_for_tabpfn(X: pd.DataFrame, split: str) -> pd.DataFrame:
    out = X.copy()

    dt_cols = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
    if dt_cols:
        print(f"[TABPFN][{split}] Dropping datetime cols: {dt_cols}")
        out = out.drop(columns=dt_cols)

    for c in out.columns:

        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
            out[c] = out[c].fillna(0.0)

        else:
            out[c] = out[c].astype(str)
            out[c] = out[c].replace("nan", "__MISSING__")

    return out


def sanity_split(name: str, X: pd.DataFrame, y: pd.Series):
    y = pd.Series(y)
    y_num = pd.to_numeric(y, errors="coerce")
    print(
        f"\n[SANITY {name}] n={len(y)} y_nan%={y.isna().mean():.4f} "
        f"y_mean={float(y_num.mean()):.6f} y_std={float(y_num.std()):.6f} "
        f"y_min={float(y_num.min()):.6f} y_max={float(y_num.max()):.6f}"
    )
    assert len(X) == len(y), f"{name}: X/y length mismatch"
    assert X.columns.duplicated().sum() == 0, f"{name}: duplicated columns in X"



@torch.no_grad()
def predict_proba_batched(clf: TabPFNClassifier, X: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
    outs = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X.iloc[i:i + batch_size]
        outs.append(clf.predict_proba(xb))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.vstack(outs) if outs else np.zeros((0, 0), dtype=np.float32)


@torch.no_grad()
def predict_batched_reg(reg: TabPFNRegressor, X: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
    outs = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X.iloc[i:i + batch_size]
        outs.append(reg.predict(xb))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)


def stratified_subsample_train(X: pd.DataFrame, y: pd.Series, max_train: int, seed: int):

    if max_train <= 0 or len(X) <= max_train:
        return X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)

    y_s = pd.Series(y)

    nunq = y_s.nunique(dropna=True)

    strat = None
    if nunq <= 50:
        counts = y_s.value_counts()
        if counts.min() >= 2:
            strat = y_s
        else:
            print("[WARN] Rare class detected → disabling stratification")

    Xs, _, ys, _ = train_test_split(
        X,
        y_s,
        train_size=max_train,
        random_state=seed,
        stratify=strat
    )

    return Xs.reset_index(drop=True), ys.reset_index(drop=True)


def _pick_numeric_cols(df: pd.DataFrame, max_cols: int = 20) -> List[str]:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cols = sorted(cols)
    return cols[:max_cols]


def _merge_asof_safe(left: pd.DataFrame, right: pd.DataFrame, *, left_on: str, right_on: str, by: str) -> pd.DataFrame:
    """
    pandas.merge_asof requirement:
      - DataFrames must be sorted by the ON key globally (not per-group),
        even when using `by=...`.
    So we always sort by [on_key, by_key] (ON first) on BOTH sides.
    """
    left2 = left[left[left_on].notna()].copy()
    right2 = right[right[right_on].notna()].copy()

    left2 = left2.sort_values([left_on, by], kind="mergesort").reset_index(drop=True)
    right2 = right2.sort_values([right_on, by], kind="mergesort").reset_index(drop=True)

    merged2 = pd.merge_asof(
        left2,
        right2,
        left_on=left_on,
        right_on=right_on,
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )

    if len(left2) != len(left):
        missing = left[left[left_on].isna()].copy()
        for c in right.columns:
            if c not in missing.columns:
                missing[c] = np.nan
        merged2 = pd.concat([merged2, missing], axis=0, ignore_index=True)

    return merged2


def _rolling_recentk_features_for_child(
    task_df: pd.DataFrame,
    child_df: pd.DataFrame,
    child_fk_col: str,
    child_time_col: str,
    child_table_obj,
    prefix: str,
    recent_k: int,
    max_numeric_cols: int = 20,
) -> pd.DataFrame:
    out_feat = pd.DataFrame(index=task_df.index)

    if recent_k <= 0:
        out_feat[f"{prefix}__cnt"] = 0
        return out_feat

    if child_fk_col not in child_df.columns or child_time_col not in child_df.columns:
        out_feat[f"{prefix}__cnt"] = 0
        return out_feat

    fk = pd.to_numeric(child_df[child_fk_col], errors="coerce")
    t = ensure_datetime(child_df[child_time_col])

    mask = fk.notna() & t.notna()
    if mask.mean() < 1.0:
        fk = fk[mask]
        t = t[mask]
        cdf = child_df.loc[mask].copy()
    else:
        cdf = child_df.copy()

    cdf["_entity_id"] = fk.astype(np.int64).values
    cdf["_child_time"] = pd.Series(t.values, index=cdf.index)

    raw = drop_pkey_fkey_time(cdf, child_table_obj)
    raw["_entity_id"] = cdf["_entity_id"].values
    raw["_child_time"] = cdf["_child_time"].values

    num_cols = _pick_numeric_cols(raw.drop(columns=["_entity_id", "_child_time"]), max_cols=max_numeric_cols)

    raw = raw[["_entity_id", "_child_time"] + num_cols].sort_values(["_entity_id", "_child_time"], kind="mergesort")

    g = raw.groupby("_entity_id", sort=False)

    raw[f"{prefix}__cnt"] = (g["_child_time"].cumcount() + 1).clip(upper=recent_k).astype(np.float32)

    if num_cols:
        roll_sum = g[num_cols].rolling(window=recent_k, min_periods=1).sum().reset_index(level=0, drop=True)
        for c in num_cols:
            raw[f"{prefix}__{c}__sum"] = roll_sum[c].astype(np.float32)
            raw[f"{prefix}__{c}__mean"] = (roll_sum[c] / raw[f"{prefix}__cnt"]).astype(np.float32)

    left = task_df[["task_row", "entity_id", "seed_time"]].copy()
    left["_entity_id"] = left["entity_id"].astype(np.int64)
    left["_seed_time"] = ensure_datetime(left["seed_time"])

    right_cols = ["_entity_id", "_child_time", f"{prefix}__cnt"]
    if num_cols:
        for c in num_cols:
            right_cols += [f"{prefix}__{c}__sum", f"{prefix}__{c}__mean"]
    right = raw[right_cols].copy()

    merged = _merge_asof_safe(left, right, left_on="_seed_time", right_on="_child_time", by="_entity_id")

    merged = merged.sort_values("task_row", kind="mergesort")

    out_feat[f"{prefix}__cnt"] = merged[f"{prefix}__cnt"].fillna(0).astype(np.float32).values
    if num_cols:
        for c in num_cols:
            out_feat[f"{prefix}__{c}__sum"] = merged[f"{prefix}__{c}__sum"].fillna(0).astype(np.float32).values
            out_feat[f"{prefix}__{c}__mean"] = merged[f"{prefix}__{c}__mean"].fillna(0).astype(np.float32).values

    return out_feat


def _rolling_recentk_features_for_hop2_via_mid(
    task_df: pd.DataFrame,
    mid_df: pd.DataFrame,
    mid_fk_to_entity: str,
    mid_time_col: str,
    mid_table_obj,
    other_df: pd.DataFrame,
    other_table_obj,
    fk2: str,
    prefix: str,
    recent_k: int,
    max_numeric_cols: int = 20,
) -> pd.DataFrame:
    out_feat = pd.DataFrame(index=task_df.index)

    if recent_k <= 0:
        out_feat[f"{prefix}__cnt"] = 0
        return out_feat

    if mid_fk_to_entity not in mid_df.columns or fk2 not in mid_df.columns or mid_time_col not in mid_df.columns:
        out_feat[f"{prefix}__cnt"] = 0
        return out_feat

    mid_fk = pd.to_numeric(mid_df[mid_fk_to_entity], errors="coerce")
    mid_t = ensure_datetime(mid_df[mid_time_col])
    other_ids = pd.to_numeric(mid_df[fk2], errors="coerce")

    mask = mid_fk.notna() & mid_t.notna() & other_ids.notna()
    if mask.mean() < 1.0:
        mdf = mid_df.loc[mask].copy()
        mid_fk = mid_fk[mask]
        mid_t = mid_t[mask]
        other_ids = other_ids[mask]
    else:
        mdf = mid_df.copy()

    mdf["_entity_id"] = mid_fk.astype(np.int64).values
    mdf["_mid_time"] = pd.Series(mid_t.values, index=mdf.index)
    mdf["_other_id"] = other_ids.astype(np.int64).values

    other_raw = drop_pkey_fkey_time(other_df, other_table_obj)
    other_num_cols = _pick_numeric_cols(other_raw, max_cols=max_numeric_cols)

    enriched = pd.DataFrame({
        "_entity_id": mdf["_entity_id"].values,
        "_mid_time": mdf["_mid_time"].values,
        "_other_id": mdf["_other_id"].values,
    })

    if other_num_cols:
        other_mat = other_raw[other_num_cols].to_numpy()
        valid = (enriched["_other_id"].values >= 0) & (enriched["_other_id"].values < other_mat.shape[0])
        other_vec = np.zeros((len(enriched), len(other_num_cols)), dtype=np.float32)
        other_vec[valid] = other_mat[enriched.loc[valid, "_other_id"].values]
        for j, c in enumerate(other_num_cols):
            enriched[c] = other_vec[:, j]

    enriched = enriched.drop(columns=["_other_id"]).sort_values(["_entity_id", "_mid_time"], kind="mergesort")
    g = enriched.groupby("_entity_id", sort=False)

    enriched[f"{prefix}__cnt"] = (g["_mid_time"].cumcount() + 1).clip(upper=recent_k).astype(np.float32)

    if other_num_cols:
        roll_sum = g[other_num_cols].rolling(window=recent_k, min_periods=1).sum().reset_index(level=0, drop=True)
        for c in other_num_cols:
            enriched[f"{prefix}__{c}__sum"] = roll_sum[c].astype(np.float32)
            enriched[f"{prefix}__{c}__mean"] = (roll_sum[c] / enriched[f"{prefix}__cnt"]).astype(np.float32)

    left = task_df[["task_row", "entity_id", "seed_time"]].copy()
    left["_entity_id"] = left["entity_id"].astype(np.int64)
    left["_seed_time"] = ensure_datetime(left["seed_time"])

    right_cols = ["_entity_id", "_mid_time", f"{prefix}__cnt"]
    if other_num_cols:
        for c in other_num_cols:
            right_cols += [f"{prefix}__{c}__sum", f"{prefix}__{c}__mean"]
    right = enriched[right_cols].copy()

    merged = _merge_asof_safe(left, right, left_on="_seed_time", right_on="_mid_time", by="_entity_id")
    merged = merged.sort_values("task_row", kind="mergesort")

    out_feat[f"{prefix}__cnt"] = merged[f"{prefix}__cnt"].fillna(0).astype(np.float32).values
    if other_num_cols:
        for c in other_num_cols:
            out_feat[f"{prefix}__{c}__sum"] = merged[f"{prefix}__{c}__sum"].fillna(0).astype(np.float32).values
            out_feat[f"{prefix}__{c}__mean"] = merged[f"{prefix}__{c}__mean"].fillna(0).astype(np.float32).values

    return out_feat


def build_features_for_split_fanout(
    db: Database,
    task: EntityTask,
    split: str,
    hop: int,
    seed: int,
    fanout_hop1: int,
    fanout_hop2: int,
    max_hop1_edges: int = 6,
    max_hop2_paths: int = 10,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    tbl = get_table_obj(task, split)
    df_task = tbl.df.copy()

    if task.entity_col not in df_task.columns:
        raise RuntimeError(f"Task table missing entity_col '{task.entity_col}'")

    entity_ids = df_task[task.entity_col].astype(int).values

    seed_time = None
    if tbl.time_col is not None and tbl.time_col in df_task.columns:
        seed_time = ensure_datetime(df_task[tbl.time_col])

    y = df_task[task.target_col].copy() if task.target_col in df_task.columns else None

    ent_table = db.table_dict[task.entity_table]
    ent_df = ent_table.df.copy()
    if ent_table.pkey_col is not None:
        assert (ent_df[ent_table.pkey_col].values == np.arange(len(ent_df))).all()

    ent_feat = drop_pkey_fkey_time(ent_df, ent_table)
    X = ent_feat.iloc[entity_ids].reset_index(drop=True)
    X.insert(0, "entity_id", entity_ids)
    if seed_time is not None:
        X.insert(1, "seed_time", seed_time.reset_index(drop=True))

    if hop == 0:
        assert y is None or len(X) == len(y), f"{split}: X/y length mismatch"
        return X, y

    if seed_time is None:
        raise RuntimeError(
            f"{split}: task table has no time_col; strict time-based split required."
        )

    task_df = pd.DataFrame({
        "task_row": np.arange(len(entity_ids), dtype=np.int64),
        "entity_id": entity_ids.astype(np.int64),
        "seed_time": seed_time.values,
    })

    incoming: List[Tuple[str, str]] = []
    for tname, t in db.table_dict.items():
        f2p = t.fkey_col_to_pkey_table or {}
        for fk_col, pk_table in f2p.items():
            if pk_table == task.entity_table:
                incoming.append((tname, fk_col))
    print(f"[HOP1 incoming-> {task.entity_table}] {incoming}")

    ent_set = set(entity_ids.tolist())
    scored = []
    for child_name, fk_col in incoming:
        cdf = db.table_dict[child_name].df
        if fk_col not in cdf.columns:
            continue
        fk_vals = pd.to_numeric(cdf[fk_col], errors="coerce").dropna().astype(np.int64)
        if len(fk_vals) == 0:
            continue
        rate = float(fk_vals.isin(ent_set).mean())
        scored.append((rate, child_name, fk_col))
    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:max_hop1_edges]

    for _, child_name, fk_col in scored:
        child_table = db.table_dict[child_name]
        cdf = child_table.df

        if child_table.time_col is None or child_table.time_col not in cdf.columns:
            raise RuntimeError(f"Child table '{child_name}' has no time_col. Strict time required.")

        feat_h1 = _rolling_recentk_features_for_child(
            task_df=task_df,
            child_df=cdf,
            child_fk_col=fk_col,
            child_time_col=child_table.time_col,
            child_table_obj=child_table,
            prefix=f"h1__{child_name}",
            recent_k=fanout_hop1,
            max_numeric_cols=20,
        )
        for c in feat_h1.columns:
            X[c] = feat_h1[c].values

        del feat_h1
        gc.collect()

    if hop == 1:
        assert y is None or len(X) == len(y), f"{split}: X/y length mismatch"
        return X, y

    paths_done = 0
    for _, mid_name, mid_fk_to_entity in scored:
        if paths_done >= max_hop2_paths:
            break

        mid_table = db.table_dict[mid_name]
        mid_df = mid_table.df
        f2p = mid_table.fkey_col_to_pkey_table or {}
        print(f"[HOP2 outgoing from {mid_name}] {list(f2p.items())}")

        if mid_table.time_col is None or mid_table.time_col not in mid_df.columns:
            raise RuntimeError(f"Mid table '{mid_name}' has no time_col. Strict time required.")

        for fk2, other_name in f2p.items():
            if paths_done >= max_hop2_paths:
                break
            if fk2 == mid_fk_to_entity:
                continue
            if other_name == task.entity_table:
                continue

            other_table = db.table_dict[other_name]
            other_df = other_table.df

            if other_table.pkey_col is None or other_table.pkey_col not in other_df.columns:
                continue
            assert (other_df[other_table.pkey_col].values == np.arange(len(other_df))).all()

            feat_h2 = _rolling_recentk_features_for_hop2_via_mid(
                task_df=task_df,
                mid_df=mid_df,
                mid_fk_to_entity=mid_fk_to_entity,
                mid_time_col=mid_table.time_col,
                mid_table_obj=mid_table,
                other_df=other_df,
                other_table_obj=other_table,
                fk2=fk2,
                prefix=f"h2__{mid_name}__{other_name}",
                recent_k=fanout_hop2,
                max_numeric_cols=20,
            )
            for c in feat_h2.columns:
                X[c] = feat_h2[c].values

            del feat_h2
            gc.collect()

            paths_done += 1

    assert y is None or len(X) == len(y), f"{split}: X/y length mismatch"
    return X, y



def train_eval_tabpfn(
    task: EntityTask,
    Xtr: pd.DataFrame,
    ytr: pd.Series,
    Xva: pd.DataFrame,
    yva: pd.Series,
    Xte: pd.DataFrame,
    yte: pd.Series,
    seed: int,
    device: str,
    model_version: str,
    max_train: int,
    pred_batch_size: int,
):
    mode = infer_mode_with_target_override(task, ytr)
    print("[FINAL MODE USED]:", mode)

    drop_cols = [c for c in ["entity_id", "seed_time"] if c in Xtr.columns]
    Xtr_m = sanitize_for_tabpfn(Xtr.drop(columns=drop_cols).copy(), "train")
    Xva_m = sanitize_for_tabpfn(Xva.drop(columns=drop_cols).copy(), "val")
    Xte_m = sanitize_for_tabpfn(Xte.drop(columns=drop_cols).copy(), "test")

    cols = list(Xtr_m.columns)
    for c in cols:
        if c not in Xva_m.columns:
            Xva_m[c] = np.nan
        if c not in Xte_m.columns:
            Xte_m[c] = np.nan
    Xva_m = Xva_m[cols]
    Xte_m = Xte_m[cols]

    print("\n===== FEATURES SHAPES (before subsample) =====")
    print(f"Xtr: {Xtr_m.shape}, Xva: {Xva_m.shape}, Xte: {Xte_m.shape}")

    ytr_num = pd.to_numeric(pd.Series(ytr), errors="coerce")
    yva_num = pd.to_numeric(pd.Series(yva), errors="coerce")
    yte_num = pd.to_numeric(pd.Series(yte), errors="coerce")
    train_mean = float(ytr_num.mean())
    val_mse_baseline = float(np.mean((np.asarray(yva_num) - train_mean) ** 2))
    test_mse_baseline = float(np.mean((np.asarray(yte_num) - train_mean) ** 2))
    print(f"\n[BASELINE] predict train-mean={train_mean:.6f}")
    print(f"[BASELINE VAL MSE]  {val_mse_baseline:.6f}")
    print(f"[BASELINE TEST MSE] {test_mse_baseline:.6f}")

    if mode == "regression":
        ytr_fit = pd.to_numeric(pd.Series(ytr), errors="coerce").astype(float)
        Xtr_fit, ytr_fit = stratified_subsample_train(Xtr_m, ytr_fit, max_train=max_train, seed=seed)
    else:
        ytr_fit = pd.Series(ytr)
        Xtr_fit, ytr_fit = stratified_subsample_train(Xtr_m, ytr_fit, max_train=max_train, seed=seed)

    print("\n===== FEATURES SHAPES (after subsample) =====")
    print(f"Xtr_fit: {Xtr_fit.shape} (max_train={max_train}), Xva: {Xva_m.shape}, Xte: {Xte_m.shape}")

    if model_version.lower() in ["v2", "2"]:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    elif model_version.lower() in ["v2.5", "2.5", "default"]:
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        reg = TabPFNRegressor(ignore_pretraining_limits=True)
    else:
        raise RuntimeError(f"Unknown model_version={model_version}. Use '2.5' or 'v2'.")

    os.environ["TABPFN_DEVICE"] = device

    t0 = time.time()

    if mode == "regression":
        yva_f = pd.to_numeric(pd.Series(yva), errors="coerce").astype(float).values
        yte_f = pd.to_numeric(pd.Series(yte), errors="coerce").astype(float).values

        if hasattr(reg, "device"):
            try:
                reg.device = device
            except Exception:
                pass

        reg.fit(Xtr_fit, np.asarray(ytr_fit, dtype=float))

        p_val = predict_batched_reg(reg, Xva_m, batch_size=max(512, pred_batch_size))
        p_test = predict_batched_reg(reg, Xte_m, batch_size=max(512, pred_batch_size))

        train_time = time.time() - t0

        mse_val = float(mean_squared_error(yva_f, p_val))
        mse_test = float(mean_squared_error(yte_f, p_test))
        print("\n===== Regression (STRICT MSE) =====")
        print(f"[VAL MSE]  {mse_val:.6f}")
        print(f"[TEST MSE] {mse_test:.6f}")

        val_tbl = get_table_obj(task, "val")
        test_tbl = get_table_obj(task, "test")
        print("[RelBench evaluate VAL]", task.evaluate(p_val, val_tbl))
        print("[RelBench evaluate TEST]", task.evaluate(p_test, test_tbl))

        return {
            "mode": mode,
            "train_time_sec": float(train_time),
            "metric_name": "mse",
            "val_metric": mse_val,
            "test_metric": mse_test,
            "extra": {"baseline_val_mse": val_mse_baseline, "baseline_test_mse": test_mse_baseline},
        }

    yva_c = pd.Series(yva)
    yte_c = pd.Series(yte)

    if hasattr(clf, "device"):
        try:
            clf.device = device
        except Exception:
            pass

    clf.fit(Xtr_fit, ytr_fit)

    proba_val = predict_proba_batched(clf, Xva_m, batch_size=pred_batch_size)
    proba_test = predict_proba_batched(clf, Xte_m, batch_size=pred_batch_size)

    train_time = time.time() - t0

    if mode == "binary":
        p_val = proba_val[:, 1]
        p_test = proba_test[:, 1]
    else:
        p_val = proba_val
        p_test = proba_test

    val_tbl = get_table_obj(task, "val")
    test_tbl = get_table_obj(task, "test")
    val_off = task.evaluate(p_val, val_tbl)
    test_off = task.evaluate(p_test, test_tbl)

    print("\n===== Classification (RelBench official) =====")
    print("[VAL]", val_off)
    print("[TEST]", test_off)

    if mode == "binary":
        yva_codes = yva_c.astype("category").cat.codes.values
        yte_codes = yte_c.astype("category").cat.codes.values
        auc_val = float(roc_auc_score(yva_codes, np.asarray(p_val)))
        auc_test = float(roc_auc_score(yte_codes, np.asarray(p_test)))
        print(f"[SKLEARN AUC] val={auc_val:.6f} test={auc_test:.6f}")
        return {
            "mode": mode,
            "train_time_sec": float(train_time),
            "metric_name": "auc",
            "val_metric": auc_val,
            "test_metric": auc_test,
            "extra": {"relbench_val": val_off, "relbench_test": test_off},
        }

    yhat_val = np.asarray(p_val).argmax(axis=1)
    yva_codes = yva_c.astype("category").cat.codes.values
    acc_val = float(accuracy_score(yva_codes, yhat_val))
    print(f"[SKLEARN ACC] val={acc_val:.6f}")
    return {
        "mode": mode,
        "train_time_sec": float(train_time),
        "metric_name": "acc",
        "val_metric": acc_val,
        "test_metric": None,
        "extra": {"relbench_val": val_off, "relbench_test": test_off},
    }


def save_results(results_root: str, dataset: str, task: str, hop: int, seed: int, payload: Dict[str, Any]):
    out_dir = osp.join(results_root, dataset, task, f"hop_{hop}", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVED] {out_dir}")


def main(
    dataset_name: str,
    task_name: str,
    hop: int,
    seed: int,
    results_root: str,
    device: str,
    model_version: str,
    max_hop1_edges: int,
    max_hop2_paths: int,
    fanout_hop1: int,
    fanout_hop2: int,
    max_train: int,
    pred_batch_size: int,
):
    script_dir = osp.dirname(osp.abspath(__file__))
    cache_dir = osp.join(script_dir, "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["RELBENCH_CACHE_DIR"] = cache_dir
    print(f"[CACHE] RELBENCH_CACHE_DIR={os.environ['RELBENCH_CACHE_DIR']}")

    np.random.seed(seed)

    dataset = get_dataset(dataset_name, download=True)
    print(f"[OK] Loaded dataset: {dataset_name}")
    try:
        print(f"[INFO] Available tasks for {dataset_name}: {get_task_names(dataset_name)}")
    except Exception:
        pass

    task = get_task(dataset_name, task_name, download=True)
    if not isinstance(task, EntityTask):
        raise RuntimeError(f"This script is for EntityTask. Got {type(task)}")

    db = dataset.get_db()

    print_first5_targets(task, "train")
    print_first5_targets(task, "val")
    print_first5_targets(task, "test")

    if hop == 0:
        print("[FANOUT] hop=0 -> [0]")
        fan1, fan2 = 0, 0
    elif hop == 1:
        print(f"[FANOUT] hop=1 -> [recentK={fanout_hop1}]")
        fan1, fan2 = fanout_hop1, 0
    else:
        print(f"[FANOUT] hop=2 -> [recentK_h1={fanout_hop1}, recentK_h2={fanout_hop2}]")
        fan1, fan2 = fanout_hop1, fanout_hop2

    Xtr, ytr = build_features_for_split_fanout(db, task, "train", hop, seed, fan1, fan2, max_hop1_edges, max_hop2_paths)
    Xva, yva = build_features_for_split_fanout(db, task, "val", hop, seed, fan1, fan2, max_hop1_edges, max_hop2_paths)
    Xte, yte = build_features_for_split_fanout(db, task, "test", hop, seed, fan1, fan2, max_hop1_edges, max_hop2_paths)

    sanity_split("train", Xtr, ytr)
    sanity_split("val", Xva, yva)
    sanity_split("test", Xte, yte)

    out = train_eval_tabpfn(
        task=task,
        Xtr=Xtr, ytr=ytr,
        Xva=Xva, yva=yva,
        Xte=Xte, yte=yte,
        seed=seed,
        device=device,
        model_version=model_version,
        max_train=max_train,
        pred_batch_size=pred_batch_size,
    )

    payload = {
        "dataset": dataset_name,
        "task": task_name,
        "hop": hop,
        "seed": seed,
        "fanout_hop1_recentK": fan1,
        "fanout_hop2_recentK": fan2,
        "task_type": str(task.task_type),
        "mode": out["mode"],
        "metric_name": out["metric_name"],
        "val_metric": out["val_metric"],
        "test_metric": out["test_metric"],
        "train_time_sec": out["train_time_sec"],
        "model": f"TabPFN({model_version})",
        "device": device,
        "max_train": max_train,
        "pred_batch_size": pred_batch_size,
        "note": "Strict-time hop features via rolling recent-K + merge_asof (ON key globally sorted).",
        "extra": out["extra"],
    }
    save_results(results_root, dataset_name, task_name, hop, seed, payload)
    print("[DONE]")


if __name__ == "__main__":
    import argparse

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("task", type=str)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hop", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model_version", type=str, default="2.5", help="Use '2.5' (default) or 'v2'")
    ap.add_argument("--results_root", type=str, default="results_tabpfn")
    ap.add_argument("--max_hop1_edges", type=int, default=6)
    ap.add_argument("--max_hop2_paths", type=int, default=10)
    ap.add_argument("--fanout_hop1", type=int, default=DEFAULT_FANOUT_HOP1)
    ap.add_argument("--fanout_hop2", type=int, default=DEFAULT_FANOUT_HOP2)
    ap.add_argument("--max_train", type=int, default=50000)
    ap.add_argument("--pred_batch_size", type=int, default=1024)

    args = ap.parse_args()

    main(
        dataset_name=args.dataset,
        task_name=args.task,
        hop=args.hop,
        seed=args.seed,
        results_root=args.results_root,
        device=args.device,
        model_version=args.model_version,
        max_hop1_edges=args.max_hop1_edges,
        max_hop2_paths=args.max_hop2_paths,
        fanout_hop1=args.fanout_hop1,
        fanout_hop2=args.fanout_hop2,
        max_train=args.max_train,
        pred_batch_size=args.pred_batch_size,
    )