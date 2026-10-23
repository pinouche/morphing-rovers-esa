import numpy as np
import torch

from morphing_rovers.autoencoder import Autoencoder, encode, load_autoencoder, save_autoencoder, train_autoencoder
from morphing_rovers.config import AutoencoderConfig
from morphing_rovers.masks import mode_views_along_straight_paths


def reconstruction_error(model: Autoencoder, views: np.ndarray) -> float:
    with torch.no_grad():
        return ((model(torch.from_numpy(views).unsqueeze(1)).squeeze(1).numpy() - views) ** 2).mean()


def test_shapes():
    model = Autoencoder(latent_dim=7, hidden_dim=16)
    views = np.random.default_rng(0).normal(size=(5, 11, 11)).astype(np.float32)
    assert model(torch.from_numpy(views).unsqueeze(1)).shape == (5, 1, 11, 11)
    assert encode(model, views).shape == (5, 7)


def test_training_reduces_the_reconstruction_error():
    rng = np.random.default_rng(0)
    patterns = rng.normal(size=(2, 11, 11))
    views = (1 + np.einsum("nk,kij->nij", rng.normal(size=(64, 2)), patterns)).astype(np.float32)
    cfg = AutoencoderConfig(latent_dim=8, hidden_dim=32, learning_rate=1e-2, batch_size=16, n_epochs=30)
    torch.manual_seed(0)
    untrained = reconstruction_error(Autoencoder(cfg.latent_dim, cfg.hidden_dim), views)
    assert reconstruction_error(train_autoencoder(views, cfg, val_fraction=0.25, seed=0), views) < 0.5 * untrained


def test_save_and_load(tmp_path):
    cfg = AutoencoderConfig(latent_dim=4, hidden_dim=8)
    model = Autoencoder(cfg.latent_dim, cfg.hidden_dim)
    save_autoencoder(model, tmp_path / "sub" / "ae.pt")
    loaded = load_autoencoder(tmp_path / "sub" / "ae.pt", cfg)
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(expected, actual)


def test_pretrained_autoencoder_reconstructs_mode_views(udp, artifacts):
    views = mode_views_along_straight_paths(udp.env, samples_per_scenario=4)
    model = load_autoencoder(artifacts / "autoencoder.pt", AutoencoderConfig())
    assert reconstruction_error(model, views) < 0.05 * ((views - views.mean(axis=0)) ** 2).mean()
