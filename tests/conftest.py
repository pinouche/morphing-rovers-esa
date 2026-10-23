from pathlib import Path

import numpy as np
import pytest

from morphing_rovers.chromosome import load_chromosome
from morphing_rovers.udp import morphing_rover_UDP


@pytest.fixture(scope="session")
def artifacts() -> Path:
    return Path(__file__).parents[1] / "artifacts"


@pytest.fixture(scope="session")
def udp() -> morphing_rover_UDP:
    """Loading the maps takes a few seconds, so the problem is shared by all tests."""
    return morphing_rover_UDP()


@pytest.fixture(scope="session")
def solution(artifacts) -> np.ndarray:
    """Submitted solution (fitness 2.0211)."""
    return load_chromosome(artifacts / "chromosomes" / "chromosome_2.0211.npy")
