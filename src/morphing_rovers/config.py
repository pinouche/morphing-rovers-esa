"""Configuration of the end-to-end optimisation (see configs/default.yml for the documented defaults)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DatasetConfig:
    samples_per_scenario: int = 1000
    val_fraction: float = 0.2


@dataclass
class AutoencoderConfig:
    checkpoint: str | None = "artifacts/autoencoder.pt"
    latent_dim: int = 50
    hidden_dim: int = 256
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.0
    batch_size: int = 512
    n_epochs: int = 1500


@dataclass
class ClusteringConfig:
    features: str = "autoencoder"
    algorithm: str = "kmeans"
    n_clusters: int = 4
    n_components: int = 3
    distance_threshold: float | None = None


@dataclass
class MaskOptimizationConfig:
    n_iter: int = 250
    learning_rate: float = 0.05
    refinement_rounds: int = 10


@dataclass
class MasksConfig:
    from_chromosome: str | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    optimization: MaskOptimizationConfig = field(default_factory=MaskOptimizationConfig)


@dataclass
class ImitationConfig:
    controller_init: str = "example"
    scenarios: list[int] | None = None
    n_runs: int = 100
    horizon_start: int = 1
    horizon_stop: int = 500
    horizon_step: int = 1
    n_iter: int = 100
    learning_rate: float = 0.002
    buffer_size: int | None = None
    radius_factors: dict[int, float] = field(default_factory=dict)
    arc_points: int = 200
    lookahead: int = 5


@dataclass
class SweepConfig:
    scenarios: list[int] = field(default_factory=lambda: [16, 19])
    radius_factors: list[float | None] = field(default_factory=lambda: [None] + [1 + 0.5 * i for i in range(18)])


@dataclass
class SubmissionConfig:
    name: str = "morphing-rovers"
    description: str = "Masks from autoencoder clustering, controller from imitation learning."


@dataclass
class Config:
    seed: int = 0
    output_dir: str = "runs/default"
    masks: MasksConfig = field(default_factory=MasksConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None, overrides: list[str] = ()) -> Config:
    """Load a YAML config (built-in defaults when `path` is None) and apply `key.subkey=value` overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Override {override!r} is not of the form key=value.")
        *parents, leaf = key.strip().split(".")
        node = data
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = yaml.safe_load(value)
    return _build(Config, data)


def _build(cls: type, data: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}. Valid keys: {sorted(names)}.")
    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        if is_dataclass(hints[name]):
            value = _build(hints[name], value or {})
        elif hints[name] is float:
            value = float(value)
        kwargs[name] = value
    return cls(**kwargs)
