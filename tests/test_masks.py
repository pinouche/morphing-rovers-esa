from math import atan2

import numpy as np
import pytest
import torch

from morphing_rovers.chromosome import split
from morphing_rovers.config import ClusteringConfig, MaskOptimizationConfig, MasksConfig
from morphing_rovers.masks import (
    build_masks,
    cluster,
    cluster_features,
    mean_best_velocity,
    mode_views_along_straight_paths,
    optimize_masks,
    relabel_by_frequency,
    velocity,
)
from morphing_rovers.udp import SCENARIO_POSITIONS, velocity_function


@pytest.fixture(scope="module")
def views(udp) -> np.ndarray:
    return mode_views_along_straight_paths(udp.env, samples_per_scenario=4)


def test_batched_velocity_matches_the_challenge_function():
    forms, views = torch.randn(3, 1, 11, 11), torch.rand(1, 5, 11, 11) + 0.5
    expected = [[velocity_function(f[0], v).item() for v in views[0]] for f in forms]
    np.testing.assert_allclose(velocity(forms, views).numpy(), expected, rtol=1e-5)


def test_mode_views_along_straight_paths(udp, views):
    assert views.shape == (30 * 5, 11, 11)
    np.testing.assert_array_equal(views[:, 5, 5], 1.0)
    start, sample = SCENARIO_POSITIONS[0][0].numpy().reshape(2, 2)
    _, first = udp.env.extract_local_view(start, atan2(*(sample - start)[::-1]), 0)
    np.testing.assert_array_equal(views[0], first.numpy())


def test_relabel_by_frequency():
    np.testing.assert_array_equal(relabel_by_frequency(np.array([5, 2, 2, 7, 2, 7])), [2, 0, 0, 1, 0, 1])


@pytest.mark.parametrize("algorithm", ["kmeans", "gmm", "agg"])
def test_cluster(algorithm):
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
    sizes = [40, 30, 20, 10]
    features = np.concatenate([c + np.random.default_rng(0).normal(0, 0.1, (n, 2)) for c, n in zip(centres, sizes, strict=True)])
    labels = cluster(features, ClusteringConfig(algorithm=algorithm), seed=0)
    np.testing.assert_array_equal(labels, np.repeat(range(4), sizes))


def test_unknown_clustering_options(views):
    with pytest.raises(ValueError, match="algorithm"):
        cluster(np.zeros((10, 2)), ClusteringConfig(algorithm="dbscan"), seed=0)
    with pytest.raises(ValueError, match="features"):
        cluster_features(views, ClusteringConfig(features="pixels"), None, seed=0)


def test_velocity_features(views):
    features = cluster_features(views, ClusteringConfig(features="velocity", n_components=3), None, seed=0)
    assert features.shape == (len(views), 3)


def test_optimize_masks_fits_each_cluster_and_pads_missing_modes():
    flat, bumpy = torch.ones(11, 11), torch.ones(11, 11) + 3 * (torch.arange(11) % 2)
    views = torch.stack([flat] * 6 + [bumpy] * 4).numpy() + np.random.default_rng(0).normal(0, 0.01, (10, 11, 11))
    labels = np.array([0] * 6 + [1] * 4)
    masks = optimize_masks(views, labels, MaskOptimizationConfig(n_iter=300, refinement_rounds=0), seed=0)
    assert masks.shape == (4, 11, 11)
    np.testing.assert_array_equal(masks[2], masks[0])
    np.testing.assert_array_equal(masks[3], masks[0])
    assert mean_best_velocity(masks[:1], views[:6]) > 0.99
    assert mean_best_velocity(masks[1:2], views[6:]) > 0.99


def test_refinement_increases_the_velocity_and_orders_masks_by_use():
    rng = np.random.default_rng(0)
    views = (1 + rng.normal(0, 1, (200, 11, 11)) * rng.uniform(0.1, 3, (200, 1, 1))).astype(np.float32)
    labels = rng.integers(0, 4, 200)
    cfg = MaskOptimizationConfig(n_iter=50, refinement_rounds=0)
    unrefined = mean_best_velocity(optimize_masks(views, labels, cfg, seed=0), views)
    cfg.refinement_rounds = 3
    masks = optimize_masks(views, labels, cfg, seed=0)
    assert mean_best_velocity(masks, views) > unrefined
    use = velocity(torch.from_numpy(masks)[:, None], torch.from_numpy(views)).argmax(dim=0).bincount(minlength=4)
    assert torch.all(use[:-1] >= use[1:])


def test_build_masks_from_chromosome(udp, solution, artifacts, tmp_path):
    cfg = MasksConfig(from_chromosome=str(artifacts / "chromosomes" / "chromosome_2.0211.npy"))
    np.testing.assert_array_equal(build_masks(cfg, udp.env, 0, tmp_path), split(solution)[0])


def test_build_masks_with_the_autoencoder(udp, artifacts, tmp_path):
    cfg = MasksConfig()
    cfg.dataset.samples_per_scenario = 4
    cfg.autoencoder.checkpoint = str(artifacts / "autoencoder.pt")
    masks = build_masks(cfg, udp.env, 0, tmp_path)
    assert masks.shape == (4, 11, 11) and np.isfinite(masks).all()
    views = mode_views_along_straight_paths(udp.env, 4)
    refined = mean_best_velocity(masks, views)
    cfg.optimization.refinement_rounds = 0
    assert refined > mean_best_velocity(build_masks(cfg, udp.env, 0, tmp_path), views)
    # The masks of the submitted solution reach 0.726 on these views.
    assert refined > 0.726
