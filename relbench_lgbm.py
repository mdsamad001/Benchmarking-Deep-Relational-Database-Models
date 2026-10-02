#!/usr/bin/env python3
import os
import gc
import json
import time
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

import lightgbm as lgb

from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names
from relbench.base import TaskType, EntityTask, Table, Database


DROP_NESTED_CATS = True

DEFAULT_FANOUT_HOP1 = 10
DEFAULT_FANOUT_HOP2 = 30

NUM_AGGS = ("mean", "sum")
CAT_AGGS = ("nunique",)


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
    tt = task.task_type
    if tt == TaskType.REGRESSION:
        return "regression"
    if tt == TaskType.BINARY_CLASSIFICATION:
        return "binary"
    if tt == TaskType.MULTICLASS_CLASSIFICATION:
        return "multiclass"
    raise RuntimeError(f"Unsupported task_type={tt}")


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


def _is_nested_obj(x) -> bool:
    return isinstance(x, (list, tuple, dict, np.ndarray))


def _col_has_nested_objects(s: pd.Series, sample: int = 2000) -> bool:
    ss = s.dropna()
    if len(ss) == 0:
        return False
    if len(ss) > sample:
        ss = ss.sample(sample, random_state=0)
    return ss.map(_is_nested_obj).any()


@dataclass
class Preproc:
    num_cols: List[str]
    cat_cols: List[str]
    num_imputer: SimpleImputer


def fit_preproc(X: pd.DataFrame, keep_cols: Optional[List[str]] = None) -> Preproc:
    df = X.copy()
    if keep_cols is not None:
        df = df.reindex(columns=keep_cols)

    num_cols = []
    cat_cols = []

    for c in df.columns:
        s = df[c]

        if s.dtype == "bool" or str(s.dtype).startswith("category"):
            cat_cols.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().any():
                num_cols.append(c)
            continue

        s_num = pd.to_numeric(s, errors="coerce")
        non_null_orig = s.notna().sum()
        non_null_num = s_num.notna().sum()

        if non_null_orig > 0 and non_null_num == non_null_orig:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    dropped_all_nan = []
    final_num_cols = []
    for c in num_cols:
        s_num = pd.to_numeric(df[c], errors="coerce")
        if s_num.notna().any():
            final_num_cols.append(c)
        else:
            dropped_all_nan.append(c)

    if dropped_all_nan:
        print(f"[INFO] Dropping all-NaN numeric columns: {dropped_all_nan}")

    num_imputer = SimpleImputer(strategy="median")
    if final_num_cols:
        df_num = df[final_num_cols].apply(pd.to_numeric, errors="coerce")
        num_imputer.fit(df_num)

    return Preproc(num_cols=final_num_cols, cat_cols=cat_cols, num_imputer=num_imputer)


def transform_preproc(
    X: pd.DataFrame, pp: Preproc, keep_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    df = X.copy()
    if keep_cols is not None:
        df = df.reindex(columns=keep_cols)

    for c in pp.num_cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in pp.cat_cols:
        if c not in df.columns:
            df[c] = np.nan

    out_parts = []

    if pp.num_cols:
        df_num_raw = df[pp.num_cols].copy()
        df_num = df_num_raw.apply(pd.to_numeric, errors="coerce")
        Xn = pp.num_imputer.transform(df_num)
        df_num = pd.DataFrame(Xn, columns=pp.num_cols, index=df.index)
        out_parts.append(df_num)

    kept_cat_cols: List[str] = []
    if pp.cat_cols:
        df_cat = df[pp.cat_cols].copy()

        bad_cols = []
        for c in df_cat.columns:
            if _col_has_nested_objects(df_cat[c]):
                bad_cols.append(c)

        if bad_cols:
            if DROP_NESTED_CATS:
                print(f"[WARN] Dropping {len(bad_cols)} categorical cols with nested objects: {bad_cols[:20]}")
                df_cat = df_cat.drop(columns=bad_cols)
            else:
                print(f"[WARN] Stringifying {len(bad_cols)} categorical cols with nested objects: {bad_cols[:20]}")
                for c in bad_cols:
                    df_cat[c] = df_cat[c].map(lambda x: "missing" if pd.isna(x) else str(x))

        kept_cat_cols = list(df_cat.columns)

        for c in df_cat.columns:
            df_cat[c] = df_cat[c].astype("string").fillna("missing").astype("category")

        if len(df_cat.columns) > 0:
            out_parts.append(df_cat)

    out = pd.concat(out_parts, axis=1) if out_parts else pd.DataFrame(index=df.index)
    return out, kept_cat_cols


def _rng_for(seed: int, split: str) -> np.random.RandomState:
    h = abs(hash((seed, split))) % (2**31 - 1)
    return np.random.RandomState(h)


def _sample_group_indices(group_sizes: np.ndarray, fanout: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Given group sizes for consecutive groups, sample up to fanout indices per group.
    Returns concatenated selected row indices (relative positions) in the concatenated group layout.
    """
    raise NotImplementedError("internal")


def _fanout_sample_by_taskrow(df: pd.DataFrame, fanout: int, rng: np.random.RandomState) -> pd.DataFrame:
    """
    df must have column 'task_row'. Fanout sample up to `fanout` rows per task_row.
    Deterministic under rng.
    """
    if fanout <= 0:
        return df.iloc[0:0].copy()

    parts = []
    for tr, g in df.groupby("task_row", sort=False):
        if len(g) <= fanout:
            parts.append(g)
        else:
            take = rng.choice(g.index.values, size=fanout, replace=False)
            parts.append(g.loc[take])
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, axis=0, ignore_index=False)


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

    rng = _rng_for(seed, split)

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

    task_rows = pd.DataFrame({
        "task_row": np.arange(len(entity_ids), dtype=np.int64),
        "entity_id": pd.Series(entity_ids, dtype="Int64"),
    })
    if seed_time is not None:
        task_rows["seed_time"] = seed_time.values

    hop1_sampled_rows_per_table: Dict[str, pd.DataFrame] = {}

    for _, child_name, fk_col in scored:
        child_table = db.table_dict[child_name]
        cdf = child_table.df.copy()

        if fk_col not in cdf.columns:
            X[f"h1__{child_name}__cnt"] = 0
            continue

        child_time = None
        if child_table.time_col is not None and child_table.time_col in cdf.columns:
            child_time = ensure_datetime(cdf[child_table.time_col])

        fk_series = pd.to_numeric(cdf[fk_col], errors="coerce")
        mask = fk_series.notna()
        cdf2 = cdf.loc[mask].copy()
        cdf2["entity_id"] = fk_series.loc[mask].astype("Int64").values

        if seed_time is not None and child_time is not None:
            cdf2["_child_time"] = child_time.loc[mask].values

        merged = cdf2.merge(task_rows, on="entity_id", how="inner")

        if seed_time is not None and child_time is not None:
            merged = merged[
                merged["_child_time"].notna()
                & merged["seed_time"].notna()
                & (merged["_child_time"] < merged["seed_time"])
            ]

        merged_s = _fanout_sample_by_taskrow(merged, fanout_hop1, rng)

        hop1_sampled_rows_per_table[child_name] = merged_s 

        feat = drop_pkey_fkey_time(merged_s, child_table)
        feat["task_row"] = merged_s["task_row"].values

        if len(feat) == 0:
            X[f"h1__{child_name}__cnt"] = 0
            del merged, merged_s, feat
            gc.collect()
            continue

        grp = feat.groupby("task_row", sort=False)
        X[f"h1__{child_name}__cnt"] = grp.size().reindex(range(len(X))).fillna(0).values

        num_cols = [c for c in feat.columns if c != "task_row" and pd.api.types.is_numeric_dtype(feat[c])]
        cat_cols = [c for c in feat.columns if c != "task_row" and (feat[c].dtype == "object" or feat[c].dtype == "bool" or str(feat[c].dtype).startswith("category"))]

        for c in num_cols:
            if "mean" in NUM_AGGS:
                X[f"h1__{child_name}__{c}__mean"] = grp[c].mean().reindex(range(len(X))).fillna(0).values
            if "sum" in NUM_AGGS:
                X[f"h1__{child_name}__{c}__sum"] = grp[c].sum().reindex(range(len(X))).fillna(0).values
            if "max" in NUM_AGGS:
                X[f"h1__{child_name}__{c}__max"] = grp[c].max().reindex(range(len(X))).fillna(0).values
            if "min" in NUM_AGGS:
                X[f"h1__{child_name}__{c}__min"] = grp[c].min().reindex(range(len(X))).fillna(0).values

        for c in cat_cols:
            if "nunique" in CAT_AGGS:
                X[f"h1__{child_name}__{c}__nunique"] = grp[c].nunique(dropna=True).reindex(range(len(X))).fillna(0).values

        del merged, merged_s, feat
        gc.collect()

    if hop == 1:
        print(f"[CHECK] split={split} entity_id head:", X["entity_id"].head(5).tolist())
        print(f"[CHECK] split={split} y head:", y.head(5).tolist() if y is not None else None)
        assert y is None or len(X) == len(y), f"{split}: X/y length mismatch"
        return X.fillna(0), y

    paths_done = 0
    for _, mid_name, mid_fk_to_entity in scored:
        mid_table = db.table_dict[mid_name]
        f2p = mid_table.fkey_col_to_pkey_table or {}
        print(f"[HOP2 outgoing from {mid_name}] {list(f2p.items())}")

        mid_sampled = hop1_sampled_rows_per_table.get(mid_name)
        if mid_sampled is None or len(mid_sampled) == 0:
            continue

        for fk2, other_name in f2p.items():
            if fk2 == mid_fk_to_entity:
                continue
            if other_name == task.entity_table:
                continue

            other_table = db.table_dict[other_name]
            other_df = other_table.df.copy()

            if other_table.pkey_col is None or other_table.pkey_col not in other_df.columns:
                continue

            assert (other_df[other_table.pkey_col].values == np.arange(len(other_df))).all()

            if fk2 not in mid_sampled.columns:
                continue

            tmp = mid_sampled[["task_row", "seed_time", fk2]].copy() if "seed_time" in mid_sampled.columns else mid_sampled[["task_row", fk2]].copy()
            tmp["_oid"] = pd.to_numeric(tmp[fk2], errors="coerce")
            tmp = tmp.dropna(subset=["_oid"]).copy()
            tmp["_oid"] = tmp["_oid"].astype(np.int64)

            if len(tmp) == 0:
                continue

            merged2 = tmp.merge(other_df, left_on="_oid", right_on=other_table.pkey_col, how="left")

            if "seed_time" in merged2.columns and other_table.time_col is not None and other_table.time_col in merged2.columns:
                merged2["_other_time"] = ensure_datetime(merged2[other_table.time_col])
                merged2 = merged2[
                    merged2["_other_time"].notna()
                    & merged2["seed_time"].notna()
                    & (merged2["_other_time"] < merged2["seed_time"])
                ]

            merged2_s = _fanout_sample_by_taskrow(merged2, fanout_hop2, rng)

            feat2 = drop_pkey_fkey_time(merged2_s, other_table)
            feat2["task_row"] = merged2_s["task_row"].values

            key_prefix = f"h2__{mid_name}__{other_name}"

            if len(feat2) == 0:
                if f"{key_prefix}__cnt" not in X.columns:
                    X[f"{key_prefix}__cnt"] = 0
                del merged2, merged2_s, feat2, tmp
                gc.collect()
                continue

            grp2 = feat2.groupby("task_row", sort=False)
            X[f"{key_prefix}__cnt"] = grp2.size().reindex(range(len(X))).fillna(0).values

            num_cols2 = [c for c in feat2.columns if c != "task_row" and pd.api.types.is_numeric_dtype(feat2[c])]
            cat_cols2 = [c for c in feat2.columns if c != "task_row" and (feat2[c].dtype == "object" or feat2[c].dtype == "bool" or str(feat2[c].dtype).startswith("category"))]

            for c in num_cols2:
                if "mean" in NUM_AGGS:
                    X[f"{key_prefix}__{c}__mean"] = grp2[c].mean().reindex(range(len(X))).fillna(0).values
                if "sum" in NUM_AGGS:
                    X[f"{key_prefix}__{c}__sum"] = grp2[c].sum().reindex(range(len(X))).fillna(0).values
                if "max" in NUM_AGGS:
                    X[f"{key_prefix}__{c}__max"] = grp2[c].max().reindex(range(len(X))).fillna(0).values
                if "min" in NUM_AGGS:
                    X[f"{key_prefix}__{c}__min"] = grp2[c].min().reindex(range(len(X))).fillna(0).values

            for c in cat_cols2:
                if "nunique" in CAT_AGGS:
                    X[f"{key_prefix}__{c}__nunique"] = grp2[c].nunique(dropna=True).reindex(range(len(X))).fillna(0).values

            del merged2, merged2_s, feat2, tmp
            gc.collect()

            paths_done += 1
            if paths_done >= max_hop2_paths:
                break

        if paths_done >= max_hop2_paths:
            break

    print(f"[CHECK] split={split} entity_id head:", X["entity_id"].head(5).tolist())
    print(f"[CHECK] split={split} y head:", y.head(5).tolist() if y is not None else None)
    assert y is None or len(X) == len(y), f"{split}: X/y length mismatch"

    return X.fillna(0), y


def train_eval(task: EntityTask, Xtr, ytr, Xva, yva, Xte, yte, cat_cols: List[str], seed: int):
    mode = infer_mode_from_task(task)

    if mode == "regression":
        params = dict(
            objective="regression",
            metric="l2",
            learning_rate=0.05,
            n_estimators=5000,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model = lgb.LGBMRegressor(**params)
    elif mode == "binary":
        params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.05,
            n_estimators=5000,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model = lgb.LGBMClassifier(**params)
    else: 
        num_class = int(pd.Series(ytr).nunique())
        params = dict(
            objective="multiclass",
            metric="multi_logloss",
            num_class=num_class,
            learning_rate=0.05,
            n_estimators=5000,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model = lgb.LGBMClassifier(**params)

    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    t0 = time.time()
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric=params["metric"],
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=callbacks,
    )
    train_time = time.time() - t0

    if mode == "regression":
        p_val = model.predict(Xva)
        p_test = model.predict(Xte)

        mse_val = float(mean_squared_error(yva, p_val))
        mse_test = float(mean_squared_error(yte, p_test))
        print("\n===== Regression (STRICT MSE) =====")
        print(f"[VAL MSE]  {mse_val:.6f}")
        print(f"[TEST MSE] {mse_test:.6f}")

        val_tbl = get_table_obj(task, "val")
        test_tbl = get_table_obj(task, "test")
        print("[RelBench evaluate VAL]", task.evaluate(p_val, val_tbl))
        print("[RelBench evaluate TEST]", task.evaluate(p_test, test_tbl))

        return {
            "mode": mode,
            "params": params,
            "train_time_sec": float(train_time),
            "metric_name": "mse",
            "val_metric": mse_val,
            "test_metric": mse_test,
            "extra": {"val_mse": mse_val, "test_mse": mse_test},
            "model": model,
        }

    proba_val = model.predict_proba(Xva)
    proba_test = model.predict_proba(Xte)
    if proba_val.ndim == 2 and proba_val.shape[1] == 2:
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
        auc_val = float(roc_auc_score(np.asarray(yva), np.asarray(p_val)))
        auc_test = float(roc_auc_score(np.asarray(yte), np.asarray(p_test)))
        print(f"[SKLEARN AUC] val={auc_val:.6f} test={auc_test:.6f}")
        return {
            "mode": mode,
            "params": params,
            "train_time_sec": float(train_time),
            "metric_name": "auc",
            "val_metric": auc_val,
            "test_metric": auc_test,
            "extra": {"val_auc": auc_val, "test_auc": auc_test, "relbench_val": val_off, "relbench_test": test_off},
            "model": model,
        }

    yhat_val = np.asarray(p_val).argmax(axis=1)
    acc_val = float(accuracy_score(np.asarray(yva), yhat_val))
    print(f"[SKLEARN ACC] val={acc_val:.6f}")
    return {
        "mode": mode,
        "params": params,
        "train_time_sec": float(train_time),
        "metric_name": "acc",
        "val_metric": acc_val,
        "test_metric": None,
        "extra": {"val_acc": acc_val, "relbench_val": val_off, "relbench_test": test_off},
        "model": model,
    }


def sanity_split(name: str, X: pd.DataFrame, y: pd.Series):
    y = pd.Series(y)
    print(
        f"\n[SANITY {name}] n={len(y)} y_nan%={y.isna().mean():.4f} "
        f"y_mean={y.mean():.4f} y_std={y.std():.4f} y_min={y.min():.4f} y_max={y.max():.4f}"
    )
    assert len(X) == len(y), f"{name}: X/y length mismatch"
    assert X.columns.duplicated().sum() == 0, f"{name}: duplicated columns in X"


def save_results(results_root: str, dataset: str, task: str, hop: int, seed: int, payload: Dict[str, Any]):
    hop_dir = osp.join(results_root, dataset, task, f"hop_{hop}")
    seed_dir = osp.join(hop_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    with open(osp.join(seed_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(osp.join(hop_dir, "summary.json"), "w") as f:
        json.dump(
            {"seed": seed, "metric_name": payload["metric_name"], "val": payload["val_metric"], "test": payload["test_metric"]},
            f,
            indent=2
        )
    print(f"[SAVED] {seed_dir}")


def main(
    dataset_name: str,
    task_name: str,
    hop: int,
    seed: int,
    datasets_dir: str,
    results_root: str,
    max_hop1_edges: int,
    max_hop2_paths: int,
    fanout_hop1: int,
    fanout_hop2: int,
):
    os.makedirs(datasets_dir, exist_ok=True)
    os.environ["RELBENCH_CACHE_DIR"] = os.path.abspath(datasets_dir)
    print(f"[CACHE] RELBENCH_CACHE_DIR={os.environ['RELBENCH_CACHE_DIR']}")

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
        print(f"[FANOUT] hop=1 -> [{fanout_hop1}]")
        fan1, fan2 = fanout_hop1, 0
    else:
        print(f"[FANOUT] hop=2 -> [{fanout_hop1},{fanout_hop2}]")
        fan1, fan2 = fanout_hop1, fanout_hop2

    Xtr, ytr = build_features_for_split_fanout(db, task, "train", hop, seed, fan1, fan2, max_hop1_edges, max_hop2_paths)
    Xva, yva = build_features_for_split_fanout(db, task, "val", hop, seed, fan1, fan2, max_hop1_edges, max_hop2_paths)
    Xte, yte = build_features_for_split_fanout(db, task, "test", hop, seed, fan1, fan2, max_hop1_edges, max_hop2_paths)

    sanity_split("train", Xtr, ytr)
    sanity_split("val", Xva, yva)
    sanity_split("test", Xte, yte)

    train_mean = float(np.mean(ytr))
    val_mse_baseline = float(np.mean((np.asarray(yva) - train_mean) ** 2))
    test_mse_baseline = float(np.mean((np.asarray(yte) - train_mean) ** 2))
    print(f"\n[BASELINE] predict train-mean={train_mean:.6f}")
    print(f"[BASELINE VAL MSE]  {val_mse_baseline:.6f}")
    print(f"[BASELINE TEST MSE] {test_mse_baseline:.6f}")

    drop_cols = [c for c in ["entity_id", "seed_time"] if c in Xtr.columns]
    keep_cols = [c for c in Xtr.columns if c not in drop_cols]

    pp = fit_preproc(Xtr, keep_cols=keep_cols)
    Xtr_lgb, cat_cols = transform_preproc(Xtr, pp, keep_cols=keep_cols)
    Xva_lgb, _ = transform_preproc(Xva, pp, keep_cols=keep_cols)
    Xte_lgb, _ = transform_preproc(Xte, pp, keep_cols=keep_cols)

    mode = infer_mode_from_task(task)
    if mode == "regression":
        ytr = ytr.astype(float)
        yva = yva.astype(float)
        yte = yte.astype(float)

    out = train_eval(task, Xtr_lgb, ytr, Xva_lgb, yva, Xte_lgb, yte, cat_cols, seed)

    payload = {
        "dataset": dataset_name,
        "task": task_name,
        "hop": hop,
        "seed": seed,
        "fanout_hop1": fan1,
        "fanout_hop2": fan2,
        "task_type": str(task.task_type),
        "mode": out["mode"],
        "metric_name": out["metric_name"],
        "val_metric": out["val_metric"],
        "test_metric": out["test_metric"],
        "train_time_sec": out["train_time_sec"],
        "params": out["params"],
        "extra": out["extra"],
        "dropped_nested_cats": bool(DROP_NESTED_CATS),
    }
    save_results(results_root, dataset_name, task_name, hop, seed, payload)
    print("[DONE]")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str)
    ap.add_argument("task", type=str)
    ap.add_argument("--hop", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets_dir", type=str, default="datasets")
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--max_hop1_edges", type=int, default=6)
    ap.add_argument("--max_hop2_paths", type=int, default=10)
    ap.add_argument("--fanout_hop1", type=int, default=DEFAULT_FANOUT_HOP1)
    ap.add_argument("--fanout_hop2", type=int, default=DEFAULT_FANOUT_HOP2)
    ap.add_argument("--keep_nested_cats", action="store_true",
                    help="If set, nested-object categorical columns are stringified instead of dropped.")
    args = ap.parse_args()

    if args.keep_nested_cats:
        DROP_NESTED_CATS = False

    main(
        dataset_name=args.dataset,
        task_name=args.task,
        hop=args.hop,
        seed=args.seed,
        datasets_dir=args.datasets_dir,
        results_root=args.results_root,
        max_hop1_edges=args.max_hop1_edges,
        max_hop2_paths=args.max_hop2_paths,
        fanout_hop1=args.fanout_hop1,
        fanout_hop2=args.fanout_hop2,
    )
