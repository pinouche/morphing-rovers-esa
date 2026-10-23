from pathlib import Path

import pytest

from morphing_rovers.config import Config, load_config

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yml"


def test_default_yaml_matches_built_in_defaults():
    assert load_config(DEFAULT_CONFIG) == Config()


def test_overrides():
    cfg = load_config(DEFAULT_CONFIG, ["imitation.n_runs=2", "imitation.scenarios=[3, 5]",
                                       "imitation.radius_factors={3: 2}", "masks.autoencoder.learning_rate=1e-3",
                                       "masks.from_chromosome=runs/x.npy", "output_dir=elsewhere"])
    assert cfg.imitation.n_runs == 2
    assert cfg.imitation.scenarios == [3, 5]
    assert cfg.imitation.radius_factors == {3: 2}
    # YAML reads "1e-3" as a string: floats are converted.
    assert cfg.masks.autoencoder.learning_rate == 1e-3
    assert cfg.masks.from_chromosome == "runs/x.npy"
    assert cfg.output_dir == "elsewhere"
    assert cfg.masks.clustering == Config().masks.clustering


@pytest.mark.parametrize("override", ["imitation.n_run=2", "typo=1", "masks.clustering.k=3"])
def test_unknown_keys_are_rejected(override):
    with pytest.raises(ValueError, match="Unknown"):
        load_config(None, [override])


def test_malformed_override():
    with pytest.raises(ValueError, match="key=value"):
        load_config(None, ["imitation.n_runs"])
