

import torch
import torch.nn as nn
from typing import Optional, List, Union
from transformers import AutoTokenizer, AutoModel
import numpy as np


from .column_embedder import ColumnEmbedder


try:
    from db_transformer.schema.columns import CategoricalColumnDef
except ImportError:

    from ...schema.columns import CategoricalColumnDef


class NomicTextEncoder(nn.Module):


    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str = 'cuda',
        trust_remote_code: bool = True
    ):
        super().__init__()

        self.device = device
        self.model_name = model_name

        print(f"Loading Nomic encoder: {model_name}...")


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        ).to(device)


        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()


        self.embed_dim = self.model.config.hidden_size

        print(f"✓ Loaded Nomic encoder (embedding dim: {self.embed_dim})")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 64
    ) -> torch.Tensor:


        all_embeddings = []


        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]


            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)


            with torch.no_grad():
                outputs = self.model(**inputs)

                batch_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(batch_embeddings)


        embeddings = torch.cat(all_embeddings, dim=0)

        return embeddings

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:


        if isinstance(texts, str):
            texts = [texts]

        return self.encode_texts(texts)


class GriffinCategoricalEmbedder(ColumnEmbedder[CategoricalColumnDef]):


    def __init__(
        self,
        target_dim: int,
        device: str = 'cuda',
        cache_embeddings: bool = True,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        shared_encoder = None,
    ):
        super().__init__()

        self.target_dim = target_dim
        self.device = device
        self.cache_embeddings = cache_embeddings


        if shared_encoder is not None:
            print(f"  ✓ Using shared Nomic encoder (memory efficient)")
            self.encoder = shared_encoder
            self.nomic_dim = self.encoder.embed_dim
        else:
            print(f"  Creating new Nomic encoder instance...")
            self.encoder = NomicTextEncoder(model_name, device)
            self.nomic_dim = self.encoder.embed_dim


        self.projection = nn.Linear(self.nomic_dim, target_dim).to(device)


        self.column_def: Optional[CategoricalColumnDef] = None
        self.category_to_text: Optional[dict] = None
        self.embedding_cache: Optional[torch.Tensor] = None

    def create(self, column_def: CategoricalColumnDef):


        self.column_def = column_def


        if hasattr(column_def, 'categories'):
            categories = column_def.categories
        elif hasattr(column_def, 'vocab'):
            categories = column_def.vocab
        else:

            categories = [f"category_{i}" for i in range(column_def.card)]

        self.category_to_text = {i: str(cat) for i, cat in enumerate(categories)}


        if self.cache_embeddings:
            self._build_embedding_cache()

    def _build_embedding_cache(self):


        if self.category_to_text is None:
            raise RuntimeError("Column definition must be set before building cache")

        print(f"Building embedding cache for {len(self.category_to_text)} categories...")


        category_texts = [
            self.category_to_text[i] 
            for i in range(len(self.category_to_text))
        ]


        with torch.no_grad():
            nomic_embeddings = self.encoder.encode_texts(category_texts)
            projected_embeddings = self.projection(nomic_embeddings)


        self.embedding_cache = projected_embeddings

        print(f"✓ Embedding cache built: {self.embedding_cache.shape}")

    def forward(self, value: torch.Tensor) -> torch.Tensor:


        if self.embedding_cache is None:
            raise RuntimeError("Embedding cache not built. Call create() first.")


        original_shape = value.shape
        indices = value.long().flatten()


        embeddings = self.embedding_cache[indices]


        embeddings = embeddings.view(*original_shape[:-1], self.target_dim)


        embeddings = embeddings.unsqueeze(-2)

        return embeddings

    def encode_new_categories(self, category_texts: List[str]) -> torch.Tensor:


        with torch.no_grad():
            nomic_embeddings = self.encoder.encode_texts(category_texts)
            projected_embeddings = self.projection(nomic_embeddings)

        return projected_embeddings

    def compute_similarity(
        self,
        category_idx_1: int,
        category_idx_2: int
    ) -> float:


        if self.embedding_cache is None:
            raise RuntimeError("Embedding cache not built")

        emb1 = self.embedding_cache[category_idx_1]
        emb2 = self.embedding_cache[category_idx_2]


        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()

        return similarity


class GriffinTextEmbedder(ColumnEmbedder):


    def __init__(
        self,
        target_dim: int,
        device: str = 'cuda',
        model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        super().__init__()

        self.target_dim = target_dim
        self.device = device


        self.encoder = NomicTextEncoder(model_name, device)
        self.nomic_dim = self.encoder.embed_dim


        self.projection = nn.Linear(self.nomic_dim, target_dim).to(device)

    def forward(self, texts: Union[str, List[str], torch.Tensor]) -> torch.Tensor:


        if isinstance(texts, torch.Tensor):


            raise NotImplementedError(
                "Text embedder expects string inputs. "
                "For categorical data, use GriffinCategoricalEmbedder."
            )

        if isinstance(texts, str):
            texts = [texts]


        with torch.no_grad():
            nomic_embeddings = self.encoder.encode_texts(texts)
            projected_embeddings = self.projection(nomic_embeddings)


        embeddings = projected_embeddings.unsqueeze(-2)

        return embeddings


def create_griffin_categorical_embedder(
    target_dim: int = 64,
    device: str = 'cuda',
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
) -> GriffinCategoricalEmbedder:


    return GriffinCategoricalEmbedder(
        target_dim=target_dim,
        device=device,
        cache_embeddings=True,
        model_name=model_name
    )