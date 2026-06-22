import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class QuantileNormalizer:
    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles
        self.quantiles   = None
        self.fitted      = False
        self.data_min    = None
        self.data_max    = None

    def fit(self, values: np.ndarray):
        values = values[np.isfinite(values)]
        if len(values) == 0:
            self.quantiles = np.array([0.0])
            self.data_min  = 0.0
            self.data_max  = 1.0
            self.fitted    = True
            return
        self.data_min = float(np.min(values))
        self.data_max = float(np.max(values))
        if self.data_min == self.data_max:
            self.quantiles = np.array([self.data_min])
            self.fitted    = True
            return
        self.quantiles = np.percentile(values, np.linspace(0, 100, self.n_quantiles))
        self.fitted    = True

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        np_values       = values.detach().cpu().numpy()
        original_shape  = np_values.shape
        np_values       = np_values.flatten()
        nan_mask        = ~np.isfinite(np_values)
        if nan_mask.any():
            np_values[nan_mask] = 0.0
        if len(self.quantiles) == 1:
            normalized = np.zeros_like(np_values)
        else:
            np_values  = np.clip(np_values, self.data_min, self.data_max)
            normalized = np.interp(np_values, self.quantiles, np.linspace(-3, 3, len(self.quantiles)))
            normalized = np.clip(normalized, -5, 5)
        return torch.tensor(normalized.reshape(original_shape), dtype=torch.float32, device=values.device)


class FloatEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FloatDecoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)


class FloatEncoderDecoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = FloatEncoder(embed_dim, hidden_dim)
        self.decoder = FloatDecoder(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        embedding     = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction

    def pretrain_step(self, batch_size: int = 256, device: str = "cuda") -> torch.Tensor:
        x = torch.randn(batch_size, 1, device=device)
        _, reconstruction = self(x)
        return torch.abs(reconstruction - x).mean()


class GriffinFloatEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_quantiles: int = 1000,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.dim        = dim
        self.hidden_dim = hidden_dim
        self.model      = FloatEncoderDecoder(dim, hidden_dim)
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
        self.normalizer          = QuantileNormalizer(n_quantiles)
        self._normalizer_fitted  = False

    def pretrain(self, num_steps=10000, lr=1e-3, batch_size=256, device="cuda", verbose=True):
        self.model.to(device).train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self.model.pretrain_step(batch_size, device)
            if torch.isnan(loss):
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if verbose and (step + 1) % 1000 == 0:
                print(f"Pre-training step {step + 1}/{num_steps}, Loss: {loss.item():.6f}")
        self.freeze_encoder_decoder()

    def freeze_encoder_decoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def fit(self, values: np.ndarray):
        if len(values) == 0:
            values = np.array([0.0])
        self.normalizer.fit(values)
        self._normalizer_fitted = True

    def save_pretrained(self, path: str):
        torch.save({"encoder": self.model.encoder.state_dict(),
                    "decoder": self.model.decoder.state_dict(),
                    "dim": self.dim, "hidden_dim": self.hidden_dim}, path)

    def load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cuda")
        self.model.encoder.load_state_dict(ckpt["encoder"])
        self.model.decoder.load_state_dict(ckpt["decoder"])
        self.freeze_encoder_decoder()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value
        self.model.encoder.eval()
        with torch.no_grad():
            embedding = self.model.encoder(normalized)
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(-2)
        return embedding


def create_pretrained_griffin_float_embedder(
    dim: int = 64,
    hidden_dim: int = 256,
    num_pretrain_steps: int = 10000,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> GriffinFloatEmbedder:
    embedder = GriffinFloatEmbedder(dim, hidden_dim)
    embedder.pretrain(num_steps=num_pretrain_steps, device=device, verbose=True)
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        embedder.save_pretrained(save_path)
    return embedder
