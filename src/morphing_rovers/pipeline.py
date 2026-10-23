"""End-to-end optimisation: rover masks from terrain clustering, then the controller by imitation learning."""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from morphing_rovers.chromosome import always_switch, assemble, load_chromosome, save_chromosome, split
from morphing_rovers.config import Config
from morphing_rovers.imitation import train_controller
from morphing_rovers.masks import build_masks
from morphing_rovers.submission import write_submission
from morphing_rovers.udp import morphing_rover_UDP

logger = logging.getLogger(__name__)


def run(cfg: Config) -> np.ndarray:
    """Optimise the masks, then the controller, and write the best chromosome and its submission file."""
    output_dir = _prepare(cfg)
    udp = morphing_rover_UDP()
    chromosome = initial_chromosome(cfg, udp, output_dir)
    best, _ = train_controller(udp, chromosome, cfg.imitation, output_dir)

    save_chromosome(output_dir / "best_chromosome.npy", best)
    (fitness,), _ = udp.pretty(best)
    write_submission(best, output_dir / "submission.json", cfg.submission.name, cfg.submission.description)
    logger.info("Best chromosome (fitness %.4f over the 30 scenarios) saved in %s", fitness, output_dir)
    return best


def sweep(cfg: Config) -> dict[int, dict[str, float]]:
    """Imitation learning on single scenarios with each candidate arc radius factor, to choose
    `imitation.radius_factors`. Results are written to sweep.json."""
    output_dir = _prepare(cfg)
    udp = morphing_rover_UDP()
    chromosome = initial_chromosome(cfg, udp, output_dir)

    results: dict[int, dict[str, float]] = {}
    best_factors = {}
    for scenario in cfg.sweep.scenarios:
        results[scenario] = {}
        for factor in cfg.sweep.radius_factors:
            imitation = replace(cfg.imitation, scenarios=[scenario], radius_factors={scenario: factor})
            _, fitness = train_controller(udp, chromosome, imitation)
            results[scenario]["straight" if factor is None else str(factor)] = fitness
            logger.info("scenario %d, radius factor %s: best fitness %.4f", scenario, factor, fitness)
        best_factors[scenario] = cfg.sweep.radius_factors[int(np.argmin(list(results[scenario].values())))]
        with open(output_dir / "sweep.json", "w") as f:
            json.dump(results, f, indent=2)
    logger.info("Best radius factor per scenario (None: straight line): %s", best_factors)
    return results


def initial_chromosome(cfg: Config, udp: morphing_rover_UDP, output_dir: Path) -> np.ndarray:
    """Optimised masks combined with the initial controller, which always switches to the best mode."""
    masks = build_masks(cfg.masks, udp.env, cfg.seed, output_dir)
    init = cfg.imitation.controller_init
    _, network = split(udp.example() if init == "example" else load_chromosome(init))
    chromosome = always_switch(assemble(masks, network))
    np.save(output_dir / "masks.npy", split(chromosome)[0])
    return chromosome


def _prepare(cfg: Config) -> Path:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    torch.manual_seed(cfg.seed)
    return output_dir
