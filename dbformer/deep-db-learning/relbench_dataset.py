

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch_frame
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

import torch_frame
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset, StatType
from torch_frame.utils import infer_series_stype

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from db_transformer.schema import columns
from db_transformer.schema.schema import (
    ColumnDef,
    ForeignKeyDef,
    Schema,
    TableSchema,
)
from db_transformer.helpers.progress import wrap_progress


class GloveTextEmbedding:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d"
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return torch.from_numpy(self.model.encode(sentences, show_progress_bar=False))


class RelBenchDBFormerDataset:


    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        data_dir: str = "./datasets",
        force_remake: bool = False,
        use_text_embeddings: bool = False,
        drop_timestamps: bool = True,
        table_dfs = None
    ):


        self.dataset_name = dataset_name
        self.task_name = task_name
        self.data_dir = data_dir
        self.use_text_embeddings = use_text_embeddings
        self.drop_timestamps = drop_timestamps

        self.relbench_dataset = get_dataset(name=dataset_name)

        self.task = get_task(dataset_name, task_name)

        self.db = self.relbench_dataset.get_db()

        self.target_table = self.task.entity_table

        self.entity_col = self.task.entity_col  
        self.target_column = 'label' 

        print(f"Loaded RelBench dataset: {dataset_name}")
        print(f"Target table: {self.target_table}, Entity column: {self.entity_col}")
        print(f"Target column (labels): {self.target_column}")
        print(f"Available tables: {list(self.db.table_dict.keys())}")
        print(f"Configuration:")
        print(f"  - Use text embeddings: {use_text_embeddings}")
        print(f"  - Drop timestamps: {drop_timestamps}")


        self.task_type_str = str(self.task.task_type)
        self.is_classification = "classification" in self.task_type_str.lower()
        self.is_regression = "regression" in self.task_type_str.lower()


        self.schema = self._build_schema_from_relbench()

    def _build_schema_from_relbench(self) -> Schema:

        schema = Schema()

        for table_name, table in self.db.table_dict.items():
            df = table.df
            column_defs = {}
            foreign_keys = []


            for col_name in df.columns:
                if table_name == self.target_table and col_name == self.target_column:

                    column_defs[col_name] = self._infer_column_def(
                        df[col_name], 
                        col_name,
                        is_target=True
                    )
                elif col_name == table.pkey_col:
                    column_defs[col_name] = columns.OmitColumnDef(key=True)
                elif col_name in table.fkey_col_to_pkey_table:
                    column_defs[col_name] = columns.OmitColumnDef(key=False)
                else:
                    column_defs[col_name] = self._infer_column_def(
                        df[col_name], 
                        col_name,
                        is_target=False
                    )


            for fkey_col, ref_table in table.fkey_col_to_pkey_table.items():
                ref_table_obj = self.db.table_dict[ref_table]
                foreign_keys.append(
                    ForeignKeyDef(
                        columns=[fkey_col],
                        ref_table=ref_table,
                        ref_columns=[ref_table_obj.pkey_col],
                    )
                )

            schema[table_name] = TableSchema(
                columns=column_defs,
                foreign_keys=foreign_keys,
            )

        return schema

    def _infer_column_def(self, series: pd.Series, col_name: str, is_target: bool = False) -> ColumnDef:

        dtype = series.dtype

        if series.apply(lambda x: isinstance(x, (np.ndarray, list, tuple))).any():
            print(f"  [SCHEMA] Column '{col_name}' is array-like; omitting from DBFormer schema.")
            return columns.OmitColumnDef(key=False)


        if is_target:
            if self.is_classification:
                nunique = series.nunique()
                return columns.CategoricalColumnDef(card=nunique, key=False)
            else:
                return columns.NumericColumnDef(key=False)


        if pd.api.types.is_numeric_dtype(dtype):
            nunique = series.nunique()
            total = len(series)
            if nunique < 100 or (total > 0 and nunique / total < 0.05):
                return columns.CategoricalColumnDef(card=nunique, key=False)
            return columns.NumericColumnDef(key=False)


        if pd.api.types.is_datetime64_any_dtype(dtype):
            return columns.DateTimeColumnDef(key=False)


        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            nunique = series.nunique()
            total = len(series)
            if total > 0 and nunique / total > 0.5:
                return columns.TextColumnDef(key=False)
            return columns.CategoricalColumnDef(card=nunique, key=False)

        return columns.OmitColumnDef(key=False)


    def build_hetero_data(
        self,
        device: str = None,
    ) -> Tuple[HeteroData, Dict[NodeType, Dict[str, Dict[StatType, Any]]]]:


        data = HeteroData()
        col_stats_dict = {}

        if self.use_text_embeddings:
            print("Using GloVe text embeddings for text columns")
            text_embedder_cfg = TextEmbedderConfig(text_embedder=GloveTextEmbedding())
        else:
            print("Treating text columns as categorical (no embeddings)")
            text_embedder_cfg = None

        table_dfs = {}
        for table_name, table in self.db.table_dict.items():
            df = table.df.copy()
            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index(drop=True)
            table_dfs[table_name] = df

        for table_name, table_schema in wrap_progress(
            self.schema.items(), verbose=True, desc="Building data"
        ):
            df = table_dfs[table_name]

            if df.empty:
                continue

            if table_name == self.target_table:
                print(f"\n{'='*60}")
                print(f"Processing TARGET table: {table_name}")
                print(f"Entity column: {self.entity_col}")
                print(f"Label column: {self.target_column}")

                try:

                    split_dfs = []
                    for split in ["train", "val", "test"]:
                        split_table = self.task.get_table(split)


                        if hasattr(split_table, "to_pandas"):
                            split_pdf = split_table.to_pandas()
                        elif hasattr(split_table, "df"):

                            split_pdf = split_table.df.copy()
                        else:
                            raise RuntimeError(
                                f"Don't know how to convert task table for split '{split}' "
                                f"to pandas; type: {type(split_table)}"
                            )

                        print(f"{split} task table shape: {split_pdf.shape}")
                        print(f"{split} task columns: {split_pdf.columns.tolist()}")
                        split_dfs.append(split_pdf)


                    task_df = pd.concat(split_dfs, ignore_index=True)


                    target_col = getattr(self.task, "target_col", None)
                    if target_col is None:
                        raise RuntimeError(
                            "RelBench task does not define target_col; cannot locate labels."
                        )

                    if self.entity_col not in task_df.columns:
                        raise RuntimeError(
                            f"Entity column '{self.entity_col}' not in task table columns: "
                            f"{task_df.columns.tolist()}"
                        )
                    if target_col not in task_df.columns:
                        raise RuntimeError(
                            f"Target column '{target_col}' not in task table columns: "
                            f"{task_df.columns.tolist()}"
                        )


                    task_df = task_df[[self.entity_col, target_col]].dropna()


                    task_df = task_df.drop_duplicates(subset=[self.entity_col], keep="last")


                    task_df = task_df.rename(columns={target_col: self.target_column})

                    print(f"Final label df shape: {task_df.shape}")
                    print(
                        f"Label distribution:\n{task_df[self.target_column].value_counts()}"
                    )


                    if self.entity_col in df.columns:
                        df = df.merge(task_df, on=self.entity_col, how="left")
                        print(f"After merge with labels - shape: {df.shape}")
                    else:
                        raise RuntimeError(
                            f"Target df columns do not contain entity_col '{self.entity_col}': "
                            f"{df.columns.tolist()}"
                        )


                    table_dfs[table_name] = df

                    self.table_dfs = table_dfs

                except Exception as e:
                    print(f"ERROR merging labels: {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Failed to merge labels for target table: {e}")

                print(f"Final DataFrame shape: {df.shape}")
                print(f"Final DataFrame columns: {df.columns.tolist()}")
                print(f"Sample data after label merge:\n{df.head()}")
                print(f"{'='*60}\n")


            if table_name == self.target_table and False:
                print(f"\n{'='*60}")
                print(f"Processing TARGET table: {table_name}")
                print(f"Target column: {self.target_column}")
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {df.columns.tolist()}")
                print(f"Sample data:\n{df.head()}")
                print(f"{'='*60}\n")


            table = self.db.table_dict[table_name]
            for fk_def in table_schema.foreign_keys:
                ref_df = table_dfs[fk_def.ref_table]
                id = (table_name, "fk-" + "-".join(fk_def.columns), fk_def.ref_table)
                try:
                    edge_index = self._fk_to_index(fk_def, df, ref_df, device)
                    data[id].edge_index = edge_index
                    if table_name == self.target_table or fk_def.ref_table == self.target_table:
                        print(f"✓ Successfully created FK edge: {id} with {edge_index.shape[1]} edges")
                except Exception as e:
                    error_detail = str(e) if str(e) else "(no error message)"
                    warnings.warn(f"Failed to join on foreign key {id}. Reason: {error_detail}")
                    if table_name == self.target_table or fk_def.ref_table == self.target_table:
                        print(f"✗ FAILED FK edge for target table: {id}")
                        print(f"  Error: {error_detail}")


            if table_name == self.target_table:

                col_to_stype = self._schema_to_stype_dict(self.schema[table_name])
                target_col = self.target_column


                if self.is_classification:
                    labels = df[target_col]
                    unique_labels = labels.dropna().unique()

                    is_effectively_integer = False
                    if pd.api.types.is_integer_dtype(labels):
                        is_effectively_integer = True
                    elif pd.api.types.is_float_dtype(labels):
                        non_nan_labels = labels.dropna()
                        if len(non_nan_labels) > 0:
                            is_effectively_integer = (non_nan_labels == non_nan_labels.astype(int)).all()

                    if is_effectively_integer:
                        print(f"  Labels are effectively integers: {sorted(unique_labels)}")
                        df[target_col] = labels.astype("Int64")


                        n_before = len(df)
                        n_nan = df[target_col].isna().sum()
                        if n_nan > 0:
                            print(f"  Filtering out {n_nan} unlabeled examples (NaN)")
                            df = df.dropna(subset=[target_col])


                            df = df.reset_index(drop=True)

                            print(f"  Rows: {n_before} → {len(df)}")
                            table_dfs[table_name] = df

                    else:
                        print(f"  Factorizing non-integer labels: {unique_labels[:10]}")
                        df[target_col], _ = pd.factorize(df[target_col])


                    col_to_stype[target_col] = torch_frame.stype.categorical
                else:

                    col_to_stype[target_col] = torch_frame.stype.numerical


                datetime_cols = [
                    c for c in df.columns
                    if c != target_col and pd.api.types.is_datetime64_any_dtype(df[c])
                ]
                if datetime_cols:
                    print(f"  Dropping datetime columns from target table: {datetime_cols}")
                    df = df.drop(columns=datetime_cols)
                    for c in datetime_cols:
                        col_to_stype.pop(c, None)
                    table_dfs[table_name] = df

            else:

                col_to_stype = self._schema_to_stype_dict(self.schema[table_name])
                target_col = None


            if self.drop_timestamps:
                cols_to_drop = []
                for col in list(df.columns):

                    if col == target_col:
                        continue


                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        cols_to_drop.append(col)
                        col_to_stype.pop(col, None)

                    elif col in col_to_stype:
                        base = getattr(col_to_stype[col], "parent", col_to_stype[col])
                        if base == torch_frame.stype.timestamp:
                            cols_to_drop.append(col)
                            col_to_stype.pop(col, None)

                if cols_to_drop:
                    print(f"  Dropping {len(cols_to_drop)} timestamp/datetime columns: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
                    df = df.drop(columns=cols_to_drop)
                    table_dfs[table_name] = df


            remapped = {}
            for col, st in col_to_stype.items():
                base = getattr(st, "parent", st)


                if base in [torch_frame.stype.embedding, torch_frame.stype.text_embedded]:
                    if self.use_text_embeddings:

                        remapped[col] = st
                    else:

                        remapped[col] = torch_frame.stype.categorical

                elif base == torch_frame.stype.timestamp:
                    if not self.drop_timestamps:

                        remapped[col] = torch_frame.stype.numerical


                else:
                    remapped[col] = st

            col_to_stype = remapped


            valid_cols = [c for c in df.columns if c in col_to_stype]
            df = df[valid_cols]


            if not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index(drop=True)


            if not self.use_text_embeddings:
                col_to_stype = {
                    c: _stype
                    for c, _stype in col_to_stype.items()
                    if getattr(_stype, 'parent', _stype) != torch_frame.stype.embedding
                }

            if len(col_to_stype) == 0 or (
                len(col_to_stype) == 1 and target_col is not None
            ):
                col_to_stype["__filler"] = torch_frame.stype.categorical
                df["__filler"] = 0


            if target_col is not None and target_col in df.columns:
                if self.is_classification:
                    col_to_stype[target_col] = torch_frame.stype.categorical
                else:
                    col_to_stype[target_col] = torch_frame.stype.numerical


            try:
                dataset = Dataset(
                    df=df,
                    col_to_stype=col_to_stype,
                    col_to_text_embedder_cfg=text_embedder_cfg,
                    target_col=target_col,
                ).materialize(device)

                print(f"Table {table_name} has stypes:")
                for stype, cols in dataset.tensor_frame.col_names_dict.items():
                    print(f"  {stype}: {cols}")

                data[table_name].tf = dataset.tensor_frame.to(device)
                col_stats_dict[table_name] = dataset.col_stats

                if table_name == self.target_table:

                    data[table_name].y = dataset.tensor_frame.y.to(device)


                    print(f"\n{'='*60}")
                    print(f"TARGET LABELS AFTER TORCH_FRAME:")
                    print(f"  y.shape: {data[table_name].y.shape}")
                    print(f"  y.dtype: {data[table_name].y.dtype}")
                    print(f"  y unique values: {torch.unique(data[table_name].y)}")
                    _y = data[table_name].y
                    _nan_count = torch.isnan(_y).sum().item() if _y.dtype.is_floating_point else 0
                    _valid = _y[~torch.isnan(_y)] if _y.dtype.is_floating_point else _y
                    _mn = _valid.min().item() if len(_valid) > 0 else float("nan")
                    _mx = _valid.max().item() if len(_valid) > 0 else float("nan")
                    print(f"  y min/max (non-NaN): {_mn:.4f}/{_mx:.4f}")
                    print(f"  NaN count: {_nan_count} / {len(_y)} rows have no label (expected for unlabeled rows)")
                    print(f"  Sample y values: {data[table_name].y[:10]}")
                    print(f"{'='*60}\n")

                    if self.is_classification:
                        data[table_name].y = data[table_name].y.long()
                    else:
                        data[table_name].y = data[table_name].y.float()

            except Exception as e:
                error_msg = f"Failed to build dataset for table {table_name}: {e}"
                if table_name == self.target_table:

                    import traceback
                    print(f"\n{'='*60}")
                    print(f"CRITICAL ERROR: Target table '{table_name}' failed to build!")
                    print(f"Error: {e}")
                    print(f"DataFrame shape: {df.shape}")
                    print(f"DataFrame columns: {df.columns.tolist()}")
                    print(f"col_to_stype: {col_to_stype}")
                    print(f"target_col: {target_col}")
                    print(f"{'='*60}")
                    traceback.print_exc()
                    raise RuntimeError(
                        f"Cannot proceed without target table. "
                        f"Target table '{table_name}' failed to build: {e}"
                    ) from e
                else:
                    warnings.warn(error_msg)
                    continue


        data: HeteroData = T.ToUndirected()(data)


        if self.target_table in data:
            print(f"\nCreating train/val/test masks for target table '{self.target_table}'...")

            try:

                def to_df(tb):
                    if isinstance(tb, pd.DataFrame):
                        return tb
                    if hasattr(tb, "df"):
                        return tb.df
                    if hasattr(tb, "to_pandas"):
                        return tb.to_pandas()
                    raise RuntimeError(
                        f"Cannot convert RelBench table of type {type(tb)} to pandas"
                    )


                train_df = to_df(self.task.get_table("train"))
                val_df   = to_df(self.task.get_table("val"))
                test_df  = to_df(self.task.get_table("test"))


                for name, df_split in [("train", train_df), ("val", val_df), ("test", test_df)]:
                    if self.entity_col not in df_split.columns:
                        raise RuntimeError(
                            f"Entity column '{self.entity_col}' not in {name}_df columns: "
                            f"{df_split.columns.tolist()}"
                        )

                train_entities = set(train_df[self.entity_col].values)
                val_entities   = set(val_df[self.entity_col].values)
                test_entities  = set(test_df[self.entity_col].values)

                print(f"  Train entities: {len(train_entities)}")
                print(f"  Val entities:   {len(val_entities)}")
                print(f"  Test entities:  {len(test_entities)}")


                target_df = table_dfs[self.target_table]
                n_total   = len(target_df)


                if self.entity_col not in target_df.columns:
                    raise RuntimeError(
                        f"Entity column '{self.entity_col}' not in target_df columns: "
                        f"{target_df.columns.tolist()}"
                    )

                train_mask = torch.zeros(n_total, dtype=torch.bool)
                val_mask   = torch.zeros(n_total, dtype=torch.bool)
                test_mask  = torch.zeros(n_total, dtype=torch.bool)

                entity_ids = target_df[self.entity_col].values
                for idx, eid in enumerate(entity_ids):
                    if eid in train_entities:
                        train_mask[idx] = True
                    elif eid in val_entities:
                        val_mask[idx] = True
                    elif eid in test_entities:
                        test_mask[idx] = True

                data[self.target_table].train_mask = train_mask.to(device)
                data[self.target_table].val_mask   = val_mask.to(device)
                data[self.target_table].test_mask  = test_mask.to(device)

                print(
                    f"  Created masks - "
                    f"Train: {train_mask.sum().item()}, "
                    f"Val: {val_mask.sum().item()}, "
                    f"Test: {test_mask.sum().item()}"
                )
                print(
                    f"  Total in any split: "
                    f"{(train_mask | val_mask | test_mask).sum().item()} / {n_total}"
                )

            except Exception as e:
                print(f"WARNING: Failed to create RelBench splits, will use random splits: {e}")
                import traceback
                traceback.print_exc()


        return data, col_stats_dict, table_dfs


    def _fk_to_index(
        self,
        fk_def: ForeignKeyDef,
        table: pd.DataFrame,
        ref_table: pd.DataFrame,
        device=None,
    ) -> torch.Tensor:

        assert isinstance(table.index, pd.RangeIndex), f"Table index must be RangeIndex, got {type(table.index)}"
        assert isinstance(ref_table.index, pd.RangeIndex), f"Ref table index must be RangeIndex, got {type(ref_table.index)}"


        fk_cols = fk_def.columns
        ref_cols = fk_def.ref_columns


        missing_fk = [c for c in fk_cols if c not in table.columns]
        missing_ref = [c for c in ref_cols if c not in ref_table.columns]

        if missing_fk:
            raise ValueError(f"FK columns {missing_fk} not found in table. Available: {table.columns.tolist()}")
        if missing_ref:
            raise ValueError(f"Ref columns {missing_ref} not found in ref table. Available: {ref_table.columns.tolist()}")


        table_subset = table[fk_cols].copy()
        ref_table_subset = ref_table[ref_cols].copy()


        table_subset['__pandas_index'] = table.index
        ref_table_subset['__pandas_index'] = ref_table.index


        out = pd.merge(
            left=table_subset,
            right=ref_table_subset,
            how="inner",
            left_on=fk_cols,
            right_on=ref_cols,
            suffixes=('_left', '_right')
        )

        if len(out) == 0:
            raise ValueError(
                f"Join produced 0 results. "
                f"FK columns: {fk_cols}, Ref columns: {ref_cols}. "
                f"Check if there are matching values."
            )

        left_idx_col = '__pandas_index_left' if '__pandas_index_left' in out.columns else '__pandas_index'
        right_idx_col = '__pandas_index_right' if '__pandas_index_right' in out.columns else '__pandas_index'

        if left_idx_col not in out.columns:
            idx_cols = [c for c in out.columns if 'pandas_index' in c]
            if len(idx_cols) >= 2:
                left_idx_col, right_idx_col = idx_cols[0], idx_cols[1]
            else:
                raise ValueError(f"Cannot find index columns in merge result. Columns: {out.columns.tolist()}")

        edge_index = torch.from_numpy(out[[left_idx_col, right_idx_col]].to_numpy()).t().contiguous()

        return edge_index.to(device) if device else edge_index

    def _schema_to_stype_dict(
        self, table_schema: TableSchema
    ) -> Dict[str, torch_frame.stype]:

        COLUMN_DEF_STYPE = {
            columns.CategoricalColumnDef: torch_frame.stype.categorical,

            columns.DateColumnDef: None,
            columns.DateTimeColumnDef: None,

            columns.TimeColumnDef: torch_frame.stype.numerical,
            columns.DurationColumnDef: torch_frame.stype.numerical,

            columns.NumericColumnDef: torch_frame.stype.numerical,

            columns.TextColumnDef: torch_frame.stype.categorical,

            columns.OmitColumnDef: None,
        }

        merged: Dict[str, torch_frame.stype] = {}
        for col_name, col_def in table_schema.columns.items():
            _stype = COLUMN_DEF_STYPE.get(type(col_def))
            if _stype is not None:
                merged[col_name] = _stype
        return merged