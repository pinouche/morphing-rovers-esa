"""End-to-end optimisation: rover masks, the routes they are refined on, then the controller by imitation learning."""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from morphing_rovers.chromosome import (
    always_switch,
    assemble,
    load_chromosome,
    reset_switch_head,
    save_chromosome,
    split,
    wired_network,
)
from morphing_rovers.config import Config
from morphing_rovers.experts import Expert, make_expert
from morphing_rovers.imitation import make_experts, train_controller, train_switching
from morphing_rovers.masks import build_masks
from morphing_rovers.routes import Route, refine_masks, schedule, search_offsets, trace_routes
from morphing_rovers.submission import write_submission
from morphing_rovers.udp import TOTAL_NUM_SCENARIOS, morphing_rover_UDP

logger = logging.getLogger(__name__)


def run(cfg: Config) -> np.ndarray:
    """Optimise the masks, then the controller, and write the best chromosome and its submission file."""
    output_dir = _prepare(cfg)
    udp = morphing_rover_UDP()
    if cfg.steering == "wired" and cfg.imitation.radius_factors:
        raise ValueError("Wired steering cannot follow arcs: use heading offsets, or `steering: learned`.")
    masks = build_masks(cfg.masks, udp.env, cfg.seed, output_dir)
    masks, experts, routes = optimise_routes(cfg, udp, masks, output_dir)
    chromosome = initial_chromosome(cfg, udp, masks, experts, output_dir)
    best, _ = train_switching(udp, chromosome, experts, routes, cfg.switching, cfg.steering, output_dir)

    save_chromosome(output_dir / "best_chromosome.npy", best)
    (fitness,), _ = udp.pretty(best)
    write_submission(best, output_dir / "submission.json", cfg.submission.name, cfg.submission.description)
    logger.info("Best chromosome (fitness %.4f over the 30 scenarios) saved in %s", fitness, output_dir)
    return best


def sweep(cfg: Config) -> dict[int, dict[str, float]]:
    """Fitness of each scenario on each candidate arc, to choose `imitation.radius_factors`, as driven by the switching
    oracle on the masks of the config. Results are written to sweep.json."""
    output_dir = _prepare(cfg)
    udp = morphing_rover_UDP()
    masks = build_masks(cfg.masks, udp.env, cfg.seed, output_dir)

    results: dict[int, dict[str, float]] = {}
    best_factors = {}
    for scenario in range(TOTAL_NUM_SCENARIOS) if cfg.sweep.scenarios is None else cfg.sweep.scenarios:
        results[scenario] = {}
        for factor in cfg.sweep.radius_factors:
            expert = make_expert(scenario, factor, cfg.imitation.arc_points, cfg.imitation.lookahead)
            fitness = schedule(masks, trace_routes(udp, masks, {scenario: expert}))[0]
            results[scenario]["straight" if factor is None else str(factor)] = fitness
            logger.info("scenario %d, radius factor %s: fitness %.4f", scenario, factor, fitness)
        best_factors[scenario] = cfg.sweep.radius_factors[int(np.argmin(list(results[scenario].values())))]
        with open(output_dir / "sweep.json", "w") as f:
            json.dump(results, f, indent=2)
    logger.info("Best radius factor per scenario (None: straight line): %s", best_factors)
    return results


def optimise_routes(
    cfg: Config, udp: morphing_rover_UDP, masks: np.ndarray, output_dir: Path
) -> tuple[np.ndarray, dict[int, Expert], dict[int, Route]]:
    """Joint optimisation of the masks, the heading offset of each scenario and the switching schedule.

    The masks are refined on the routes of the experts, then the search for the fastest offset of each scenario and
    the refinement of the masks on the resulting routes alternate. Returns the masks, experts and routes, and saves
    the offsets to heading_offsets.json.
    """
    imitation = cfg.imitation
    experts = make_experts(imitation)
    routes = trace_routes(udp, masks, experts)
    masks = refine_masks(masks, routes, cfg.masks.schedule)
    for round_id in range(1, cfg.offsets.rounds + 1):
        logger.info("Heading offsets, round %d of %d", round_id, cfg.offsets.rounds)
        imitation = replace(imitation, heading_offsets=search_offsets(udp, masks, experts, cfg.offsets.candidates))
        experts = make_experts(imitation)
        routes = trace_routes(udp, masks, experts)
        masks = refine_masks(masks, routes, cfg.masks.schedule)
    with open(output_dir / "heading_offsets.json", "w") as f:
        json.dump(imitation.heading_offsets, f, indent=2)
    return masks, experts, routes


def initial_chromosome(
    cfg: Config, udp: morphing_rover_UDP, masks: np.ndarray, experts: dict[int, Expert], output_dir: Path
) -> np.ndarray:
    """The masks with a controller that steers but is yet to learn when to morph, saved as masks.npy."""
    if cfg.steering == "wired":
        chromosome = assemble(masks, wired_network())
    elif cfg.steering == "learned":
        init = cfg.imitation.controller_init
        _, network = split(udp.example() if init == "example" else load_chromosome(init))
        steering, _ = train_controller(udp, always_switch(assemble(masks, network)), cfg.imitation, experts, output_dir)
        chromosome = reset_switch_head(steering)
    else:
        raise ValueError(f"Unknown steering {cfg.steering!r}, expected 'wired' or 'learned'.")
    np.save(output_dir / "masks.npy", split(chromosome)[0])
    return chromosome


def _prepare(cfg: Config) -> Path:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    torch.manual_seed(cfg.seed)
    return output_dir
