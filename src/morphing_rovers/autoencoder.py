"""Convolutional autoencoder of the 11x11 terrain patches ("mode views") a rover stands on.

Its latent space is used to cluster the terrain into one type per rover mode.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from morphing_rovers.config import AutoencoderConfig
from morphing_rovers.udp import MASK_SIZE

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_lin(self.flatten(self.encoder_cnn(x)))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(True),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 1, 1))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=0),
        )
        self.linear_output = nn.Linear(MASK_SIZE**2, MASK_SIZE**2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_conv(self.unflatten(self.decoder_lin(x)))
        x = self.linear_output(torch.flatten(x, start_dim=1))
        return x.reshape(-1, 1, MASK_SIZE, MASK_SIZE)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(views: np.ndarray, cfg: AutoencoderConfig, val_fraction: float, seed: int) -> Autoencoder:
    """Train on mode views (N, 11, 11) and return the model with the best validation loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(seed)
    data = torch.from_numpy(np.asarray(views, dtype=np.float32)).unsqueeze(1)
    data = data[torch.randperm(len(data), generator=generator)]
    n_val = int(val_fraction * len(data))
    val, train = data[:n_val].to(device), data[n_val:]
    loader = DataLoader(TensorDataset(train), cfg.batch_size, shuffle=True, generator=generator)

    model = Autoencoder(cfg.latent_dim, cfg.hidden_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for epoch in range(cfg.n_epochs):
        model.train()
        train_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            loss = F.mse_loss(model(x), x)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() / len(loader)
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(val), val).item() if n_val else train_loss
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        if (epoch + 1) % 50 == 0 or epoch + 1 == cfg.n_epochs:
            logger.info("autoencoder epoch %d: train loss %.5f, val loss %.5f", epoch + 1, train_loss, val_loss)
    model.load_state_dict(best_state)
    return model.cpu().eval()


def save_autoencoder(model: Autoencoder, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


def load_autoencoder(path: str | Path, cfg: AutoencoderConfig) -> Autoencoder:
    model = Autoencoder(cfg.latent_dim, cfg.hidden_dim)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.eval()


def encode(model: Autoencoder, views: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Latent representation (N, latent_dim) of mode views (N, 11, 11)."""
    data = torch.from_numpy(np.asarray(views, dtype=np.float32)).unsqueeze(1)
    with torch.no_grad():
        return torch.cat([model.encoder(batch) for batch in data.split(batch_size)]).numpy()
