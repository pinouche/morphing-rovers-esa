"""Split, assemble, save and load chromosomes (decision vectors).

A chromosome is [4 masks of 11x11 | network biases | network weights | 7 network hyperparameters].
"""

import pickle
from pathlib import Path

import numpy as np

from morphing_rovers.udp import (
    FLOAT_MAX,
    MASK_SIZE,
    NETWORK_SETUP,
    NUM_BIASES,
    NUM_MODE_PARAMETERS,
    NUM_NN_PARAMS,
    NUMBER_OF_MODES,
    get_bounds,
)

MASKS_SHAPE = (NUMBER_OF_MODES, MASK_SIZE, MASK_SIZE)
CHROMOSOME_SIZE = NUM_MODE_PARAMETERS + NUM_NN_PARAMS + 7

# The network outputs are [mode switch command, angular change]: their biases are the last two biases
# and their weights (shape [2, hidden]) the last weights of the network.
_HIDDEN = NETWORK_SETUP["hidden_neurons"][1]
SWITCH_BIAS = NUM_MODE_PARAMETERS + NUM_BIASES - 2
SWITCH_WEIGHTS = slice(NUM_MODE_PARAMETERS + NUM_NN_PARAMS - 2 * _HIDDEN, NUM_MODE_PARAMETERS + NUM_NN_PARAMS - _HIDDEN)


def split(chromosome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the masks, shaped (4, 11, 11), and the network part of a chromosome."""
    chromosome = np.asarray(chromosome, dtype=np.float64)
    return chromosome[:NUM_MODE_PARAMETERS].reshape(MASKS_SHAPE), chromosome[NUM_MODE_PARAMETERS:]


def assemble(masks: np.ndarray, network: np.ndarray) -> np.ndarray:
    """Concatenate masks and network parameters into a chromosome that lies within the problem bounds."""
    chromosome = np.concatenate([np.asarray(masks, dtype=np.float64).ravel(), np.asarray(network, dtype=np.float64)])
    if chromosome.shape != (CHROMOSOME_SIZE,):
        raise ValueError(f"Chromosome has {chromosome.size} values, expected {CHROMOSOME_SIZE}.")
    return clip_to_bounds(chromosome)


def always_switch(chromosome: np.ndarray) -> np.ndarray:
    """Make the rover switch to the best mode whenever its cooldown allows it.

    The mode switch command becomes a constant positive bias, which keeps the chromosome within bounds
    (unlike the bias of 10000 used previously).
    """
    chromosome = np.array(chromosome, dtype=np.float64)
    chromosome[SWITCH_WEIGHTS] = 0.0
    chromosome[SWITCH_BIAS] = FLOAT_MAX
    return chromosome


def clip_to_bounds(chromosome: np.ndarray) -> np.ndarray:
    lower, upper = get_bounds()
    return np.clip(chromosome, lower, upper)


def load_chromosome(path: str | Path) -> np.ndarray:
    """Load a chromosome saved with `save_chromosome`, or pickled by the previous version of this code."""
    path = Path(path)
    if path.suffix == ".npy":
        chromosome = np.load(path)
    else:
        with open(path, "rb") as f:
            chromosome = pickle.load(f)
    return np.asarray(chromosome, dtype=np.float64)


def save_chromosome(path: str | Path, chromosome: np.ndarray) -> None:
    np.save(path, np.asarray(chromosome, dtype=np.float64))
