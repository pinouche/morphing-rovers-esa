"""Submission file for https://optimise.esa.int/challenge/spoc-2-morphing-rovers."""

import json
from pathlib import Path

import numpy as np

from morphing_rovers.udp import get_bounds

CHALLENGE_ID = "spoc-2-morphing-rovers"
PROBLEM_ID = "morphing-rovers"


def write_submission(chromosome: np.ndarray, path: str | Path, name: str = "", description: str = "") -> None:
    """Write a chromosome in the JSON format of the ESA optimise platform, after checking that it is valid."""
    x = np.asarray(chromosome, dtype=np.float64)
    lower, upper = (np.asarray(bound, dtype=np.float64) for bound in get_bounds())
    if x.shape != lower.shape:
        raise ValueError(f"Chromosome has shape {x.shape}, expected {lower.shape}.")
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("Chromosome is out of bounds; clip it with morphing_rovers.chromosome.clip_to_bounds.")
    if np.any(x[-7:] != np.round(x[-7:])):
        raise ValueError("The last 7 values (pooling and activation types) must be integers.")

    submission = {"decisionVector": x.tolist(), "problem": PROBLEM_ID, "challenge": CHALLENGE_ID,
                  "name": name, "description": description}
    with open(path, "w") as f:
        json.dump([submission], f, indent=6)
