

from copy import deepcopy
from typing import Any, Dict, List, Optional
import warnings
import torch
import torch.nn as nn
from torch_geometric.typing import NodeType

from torch_frame import stype, TensorFrame
from torch_frame.data import StatType

from .columns.griffin_categorical_embedder import GriffinCategoricalEmbedder, NomicTextEncoder
from .columns.griffin_float_embedder import GriffinFloatEmbedder


class GriffinTableEmbedder(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        col_stats: Dict[str, Dict[StatType, Any]],
        col_names_dict: Dict[stype, List[str]],
        device: str = 'cuda',
        pretrained_float_path: Optional[str] = None,
        shared_nomic_encoder = None,
        category_mappings: Optional[Dict[str, List[str]]] = None,
        table_dataframes: Optional[Dict[NodeType, "pd.DataFrame"]] = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.col_names_dict = col_names_dict
        self.col_stats = col_stats
        self.device = device


        self._active_stypes = {
            _stype: col_names_dict.get(_stype, [])
            for _stype in [stype.categorical, stype.numerical]
            if _stype in col_names_dict and len(col_names_dict[_stype]) > 0
        }


        self._active_cols = []
        for _stype in self._active_stypes:
            self._active_cols.extend(col_names_dict.get(_stype, []))


        self.cat_embedder = None
        self.num_embedder = None


        if stype.categorical in self._active_stypes:
            self.cat_embedder = GriffinCategoricalEmbedder(
                target_dim=embed_dim,
                device=device,
                cache_embeddings=True,
                shared_encoder=shared_nomic_encoder,
            )
            print(f"✓ Initialized Griffin categorical embedder for table")


            if category_mappings:
                self._build_real_cache(category_mappings)
            else:

                print(f"  WARNING: No category mappings provided, using dummy cache")
                self._build_dummy_cache()


        if stype.numerical in self._active_stypes:
            self.num_embedder = GriffinFloatEmbedder(
                dim=embed_dim,
                pretrained_path=pretrained_float_path,
            )
            print(f"✓ Initialized Griffin numerical embedder for table")

    def _build_real_cache(self, category_mappings: Dict[str, List[str]]):

        print(f"  Building cache with REAL category names...")


        all_categories = []
        cat_cols = self.col_names_dict.get(stype.categorical, [])

        for col_name in cat_cols:
            if col_name in category_mappings:

                categories = category_mappings[col_name]
                all_categories.extend(categories)


        unique_categories = list(dict.fromkeys(all_categories))

        if not unique_categories:
            print(f"  WARNING: No categories found, using fallback")
            self._build_dummy_cache()
            return

        print(f"  Found {len(unique_categories)} unique categories")
        print(f"  Sample categories: {unique_categories[:5]}")


        from types import SimpleNamespace
        col_def = SimpleNamespace(
            card=len(unique_categories),
            categories=unique_categories
        )


        self.cat_embedder.create(col_def)
        print(f"  ✓ Cache built with {len(unique_categories)} REAL categories")

    def _build_dummy_cache(self):

        cache_size = 10000
        print(f"  Using cache size: {cache_size} dummy categories")

        from types import SimpleNamespace
        col_def = SimpleNamespace(
            card=cache_size,
            categories=[f"cat_{i}" for i in range(cache_size)]
        )

        self.cat_embedder.create(col_def)
        print(f"  ✓ Embedding cache built ({cache_size} categories)")

    @property
    def active_stypes(self) -> Dict[stype, List[str]]:
        return self._active_stypes

    @property
    def active_cols(self) -> List[str]:
        return self._active_cols

    def forward(self, tf: TensorFrame) -> torch.Tensor:

        xs = []


        if stype.categorical in tf.feat_dict and self.cat_embedder is not None:
            cat_feat = tf.feat_dict[stype.categorical]
            num_cat_cols = cat_feat.shape[1]

            for col_idx in range(num_cat_cols):
                col_values = cat_feat[:, col_idx:col_idx+1]

                try:
                    col_emb = self.cat_embedder(col_values)
                    xs.append(col_emb)
                except Exception as e:
                    warnings.warn(
                        f"Failed to embed categorical column {col_idx}: {e}. "
                        f"Using zero embedding."
                    )
                    zero_emb = torch.zeros(
                        cat_feat.shape[0], 1, self.embed_dim,
                        device=self.device
                    )
                    xs.append(zero_emb)


        if stype.numerical in tf.feat_dict and self.num_embedder is not None:
            num_feat = tf.feat_dict[stype.numerical]
            num_num_cols = num_feat.shape[1]

            for col_idx in range(num_num_cols):
                col_values = num_feat[:, col_idx:col_idx+1]

                try:
                    col_emb = self.num_embedder(col_values)
                    xs.append(col_emb)
                except Exception as e:
                    warnings.warn(
                        f"Failed to embed numerical column {col_idx}: {e}. "
                        f"Using zero embedding."
                    )
                    zero_emb = torch.zeros(
                        num_feat.shape[0], 1, self.embed_dim,
                        device=self.device
                    )
                    xs.append(zero_emb)

        if len(xs) == 0:
            x = torch.zeros(
                (tf.num_rows, 1, self.embed_dim),
                dtype=torch.float32,
                device=self.device
            )
        else:
            x = torch.cat(xs, dim=1)

        return x


class GriffinDBEmbedder(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        col_names_dict_per_table: Dict[NodeType, Dict[stype, List[str]]],
        device: str = 'cuda',
        pretrained_float_path: Optional[str] = None,
        table_dataframes: Optional[Dict[NodeType, 'pd.DataFrame']] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.device = device

        print(f"\n{'='*60}")
        print(f"Initializing GriffinDBEmbedder")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Device: {device}")
        print(f"  Tables: {len(col_names_dict_per_table)}")
        print(f"  Using REAL category names: {table_dataframes is not None}")
        print(f"{'='*60}\n")


        category_mappings_per_table = {}
        if table_dataframes:
            print("Extracting REAL category names from data...")
            for table_name, df in table_dataframes.items():
                if table_name not in col_names_dict_per_table:
                    continue

                cat_cols = col_names_dict_per_table[table_name].get(stype.categorical, [])
                if not cat_cols:
                    continue

                table_mappings = {}
                for col_name in cat_cols:
                    if col_name in df.columns:

                        unique_vals = df[col_name].dropna().unique().astype(str).tolist()
                        table_mappings[col_name] = unique_vals
                        print(f"  {table_name}.{col_name}: {len(unique_vals)} categories")

                if table_mappings:
                    category_mappings_per_table[table_name] = table_mappings

            print(f"✓ Extracted category names from {len(category_mappings_per_table)} tables\n")


        needs_nomic = any(
            stype.categorical in col_names_dict_per_table.get(table, {})
            for table in col_names_dict_per_table
        )

        if needs_nomic:
            print("Creating shared Nomic encoder...")
            self.shared_nomic_encoder = NomicTextEncoder("nomic-ai/nomic-embed-text-v1.5", device)
            print(f"✓ Shared Nomic encoder loaded\n")
        else:
            self.shared_nomic_encoder = None


        self.active_cols_dict = {}
        self.active_stypes_dict = {}


        self.embedders = nn.ModuleDict()

        for table_name in col_names_dict_per_table:
            print(f"Creating embedder for table: {table_name}")


            table_cat_mappings = category_mappings_per_table.get(table_name, None)

            embedder = GriffinTableEmbedder(
                embed_dim=embed_dim,
                col_stats=col_stats_dict[table_name],
                col_names_dict=col_names_dict_per_table[table_name],
                device=device,
                pretrained_float_path=pretrained_float_path,
                shared_nomic_encoder=self.shared_nomic_encoder,
                category_mappings=table_cat_mappings,
                table_dataframes = table_dataframes
            )

            self.embedders[table_name] = embedder
            self.active_cols_dict[table_name] = embedder.active_cols
            self.active_stypes_dict[table_name] = embedder.active_stypes

            if len(embedder.active_cols) == 0:
                self.active_cols_dict[table_name] = ["__filler"]

        print(f"\n✓ GriffinDBEmbedder initialized successfully\n")

    def fit_normalizers(self, train_tf_dict: Dict[NodeType, TensorFrame]):

        import numpy as np

        print("\nFitting Griffin normalizers...")
        print("Fitting numerical normalizers on training data...")

        for table_name, tf in train_tf_dict.items():
            if table_name not in self.embedders:
                continue

            embedder = self.embedders[table_name]

            if stype.numerical in tf.feat_dict and embedder.num_embedder is not None:
                num_data = tf.feat_dict[stype.numerical]
                values_np = num_data.detach().cpu().numpy().flatten()
                embedder.num_embedder.fit(values_np)
                print(f"  ✓ Fitted numerical normalizer on {len(values_np)} training values")

        print("✓ All numerical normalizers fitted\n")

    def forward(
        self,
        tf_dict: Dict[NodeType, TensorFrame]
    ) -> Dict[NodeType, torch.Tensor]:

        x_dict = {}

        for table_name, tf in tf_dict.items():
            if table_name not in self.embedders:
                warnings.warn(f"No embedder found for table {table_name}, skipping")
                continue

            try:
                x_dict[table_name] = self.embedders[table_name](tf)
            except Exception as e:
                warnings.warn(f"Failed to embed table {table_name}: {e}")
                x_dict[table_name] = torch.zeros(
                    (tf.num_rows, 1, self.embed_dim),
                    dtype=torch.float32,
                    device=self.device
                )

        return x_dict