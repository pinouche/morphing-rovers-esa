"""Rover mode masks: cluster the terrain met along the start-to-sample lines, then fit one mask per cluster."""

import logging
from math import atan2
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from morphing_rovers.autoencoder import Autoencoder, encode, load_autoencoder, save_autoencoder, train_autoencoder
from morphing_rovers.chromosome import load_chromosome, split
from morphing_rovers.config import ClusteringConfig, MaskOptimizationConfig, MasksConfig
from morphing_rovers.udp import (
    EPS_C,
    MASK_SIZE,
    NUMBER_OF_MODES,
    SCENARIO_POSITIONS,
    MysteriousMars,
    distance_to_velocity,
    scenario_ids,
)

logger = logging.getLogger(__name__)

# Number of reference views used to describe each view by its velocity profile (`features: velocity`).
N_LANDMARKS = 300


def mode_views_along_straight_paths(env: MysteriousMars, samples_per_scenario: int) -> np.ndarray:
    """Mode views (N, 11, 11) seen by a rover driving in a straight line from each landing site to its sample."""
    views = []
    for map_id, scenario_id in scenario_ids():
        start, sample = SCENARIO_POSITIONS[map_id][scenario_id].numpy().reshape(2, 2)
        heading = atan2(sample[1] - start[1], sample[0] - start[0])
        for t in np.linspace(0, 1, samples_per_scenario + 1):
            _, mode_view = env.extract_local_view((1 - t) * start + t * sample, heading, map_id)
            views.append(mode_view.numpy())
    return np.stack(views)


def velocity(forms: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
    """Batched `udp.velocity_function`, broadcasting over the leading dimensions of (..., 11, 11) tensors."""
    f_norm = forms.norm(dim=(-2, -1))
    mv_norm = views.norm(dim=(-2, -1))
    luminance = (2 * f_norm * mv_norm + EPS_C) / (f_norm**2 + mv_norm**2 + EPS_C)
    distance = (2 - (forms / f_norm[..., None, None] - views / mv_norm[..., None, None]).norm(dim=(-2, -1))) * 0.5
    metric = (1 - distance * luminance.sqrt()) * 2
    return distance_to_velocity(metric)


def mean_best_velocity(masks: np.ndarray, views: np.ndarray) -> float:
    """Average velocity factor over the views when the rover always uses its best mode."""
    v = velocity(torch.as_tensor(masks, dtype=torch.float32)[:, None], torch.as_tensor(views, dtype=torch.float32))
    return v.max(dim=0).values.mean().item()


def cluster_features(
    views: np.ndarray, cfg: ClusteringConfig, autoencoder: Autoencoder | None, seed: int
) -> np.ndarray:
    """Low-dimensional description of each view, weighted by the explained variance of its PCA components."""
    if cfg.features == "autoencoder":
        raw = encode(autoencoder, views)
    elif cfg.features == "velocity":
        # How fast a rover whose mask is the view itself would drive over a set of reference views.
        landmarks = views[np.random.default_rng(seed).permutation(len(views))[:N_LANDMARKS]]
        forms, landmarks = torch.as_tensor(views)[:, None], torch.as_tensor(landmarks)[None]
        raw = torch.cat([velocity(chunk, landmarks) for chunk in forms.split(1000)]).numpy()
    else:
        raise ValueError(f"Unknown clustering features {cfg.features!r}, expected 'autoencoder' or 'velocity'.")
    pca = PCA(n_components=cfg.n_components).fit(raw)
    return pca.transform(raw) * pca.explained_variance_ratio_


def cluster(features: np.ndarray, cfg: ClusteringConfig, seed: int) -> np.ndarray:
    """Cluster labels, renumbered so that cluster 0 is the largest (the rover starts in mode 0)."""
    if cfg.algorithm == "kmeans":
        labels = KMeans(n_clusters=cfg.n_clusters, random_state=seed, n_init="auto").fit_predict(features)
    elif cfg.algorithm == "gmm":
        model = GaussianMixture(n_components=cfg.n_clusters, init_params="k-means++", random_state=seed)
        labels = model.fit(features).predict(features)
    elif cfg.algorithm == "agg":
        n_clusters = None if cfg.distance_threshold is not None else cfg.n_clusters
        model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=cfg.distance_threshold,
                                        linkage="average")
        labels = model.fit_predict(features)
    else:
        raise ValueError(f"Unknown clustering algorithm {cfg.algorithm!r}, expected 'kmeans', 'gmm' or 'agg'.")
    return relabel_by_frequency(labels)


def relabel_by_frequency(labels: np.ndarray) -> np.ndarray:
    values, counts = np.unique(labels, return_counts=True)
    rank = {value: i for i, value in enumerate(values[np.argsort(-counts, kind="stable")])}
    return np.array([rank[label] for label in labels])


def fit_mask(views: torch.Tensor, mask: torch.Tensor, cfg: MaskOptimizationConfig) -> torch.Tensor:
    """Gradient ascent on the mean velocity of a mask over the given views, starting from `mask`."""
    mask = mask.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([mask], cfg.learning_rate)
    for _ in range(cfg.n_iter):
        loss = -velocity(mask, views).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return mask.detach()


def optimize_masks(views: np.ndarray, labels: np.ndarray, cfg: MaskOptimizationConfig, seed: int) -> np.ndarray:
    """One mask per cluster for the (up to 4) largest clusters; missing modes start as copies of the first mask.

    The masks are then refined by alternately assigning each view to its fastest mask (the mode the rover
    switches to) and refitting each mask on its views, i.e. k-means with the velocity as similarity. The masks
    are returned by decreasing use, as the rover starts in the first mode.
    """
    generator = torch.Generator().manual_seed(seed)
    views = torch.as_tensor(views, dtype=torch.float32)
    masks = [fit_mask(views[labels == i], torch.rand(MASK_SIZE, MASK_SIZE, generator=generator), cfg)
             for i in range(NUMBER_OF_MODES) if np.any(labels == i)]
    masks = torch.stack(masks + [masks[0]] * (NUMBER_OF_MODES - len(masks)))
    for _ in range(cfg.refinement_rounds):
        fastest = velocity(masks[:, None], views).argmax(dim=0)
        masks = torch.stack([fit_mask(views[fastest == i], masks[i], cfg) if torch.any(fastest == i) else masks[i]
                             for i in range(NUMBER_OF_MODES)])
    use = velocity(masks[:, None], views).argmax(dim=0).bincount(minlength=NUMBER_OF_MODES)
    return masks[torch.argsort(use, descending=True, stable=True)].numpy()


def build_masks(cfg: MasksConfig, env: MysteriousMars, seed: int, output_dir: Path) -> np.ndarray:
    """Masks (4, 11, 11), either taken from an existing chromosome or optimised from the terrain."""
    if cfg.from_chromosome:
        logger.info("Using the masks of %s", cfg.from_chromosome)
        return split(load_chromosome(cfg.from_chromosome))[0]

    views = mode_views_along_straight_paths(env, cfg.dataset.samples_per_scenario)
    autoencoder = None
    if cfg.clustering.features == "autoencoder":
        autoencoder = _load_or_train_autoencoder(cfg, views, seed, output_dir)
    labels = cluster(cluster_features(views, cfg.clustering, autoencoder, seed), cfg.clustering, seed)
    masks = optimize_masks(views, labels, cfg.optimization, seed)
    logger.info("Cluster sizes %s, mean velocity factor along the straight paths: %.4f",
                np.bincount(labels).tolist(), mean_best_velocity(masks, views))
    return masks


def _load_or_train_autoencoder(cfg: MasksConfig, views: np.ndarray, seed: int, output_dir: Path) -> Autoencoder:
    path = Path(cfg.autoencoder.checkpoint) if cfg.autoencoder.checkpoint else output_dir / "autoencoder.pt"
    if path.exists():
        logger.info("Loading autoencoder %s", path)
        return load_autoencoder(path, cfg.autoencoder)
    logger.info("Training autoencoder on %d mode views", len(views))
    model = train_autoencoder(views, cfg.autoencoder, cfg.dataset.val_fraction, seed)
    save_autoencoder(model, path)
    logger.info("Saved autoencoder to %s", path)
    return model
