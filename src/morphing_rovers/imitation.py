"""Imitation learning of the rover steering (DAgger-style).

The rover drives with its current controller while an expert tells, at every visited state, how it should have
turned. The controller is fitted to these targets, then the number of recorded steps per scenario (the horizon)
grows, so that the controller learns to correct its own mistakes along the whole trajectory.
"""

import csv
import logging
from pathlib import Path

import numpy as np
import torch

from morphing_rovers.chromosome import save_chromosome
from morphing_rovers.config import ImitationConfig
from morphing_rovers.experts import Expert, make_expert
from morphing_rovers.udp import (
    FLOAT_MAX,
    FLOAT_MIN,
    MAX_DA,
    SCENARIOS_PER_MAP,
    TOTAL_NUM_SCENARIOS,
    Controller,
    Rover,
    morphing_rover_UDP,
)

logger = logging.getLogger(__name__)

# (rover view, rover state, latent state, expert orientation change): the network inputs and its target.
Sample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]


class _Recorder:
    """`on_step` hook of `run_single_scenario` recording the first `horizon` steps of a scenario."""

    def __init__(self, expert: Expert, horizon: int, samples: list[Sample]):
        self.expert = expert
        self.horizon = horizon
        self.samples = samples
        self.steps = 0

    def __call__(self, rover: Rover, rover_view, distance_vector, original_distance) -> None:
        if self.steps == self.horizon:
            return
        self.steps += 1
        target = self.expert.steering(rover.position.numpy(), rover.angle)
        state = rover.network_state(distance_vector, original_distance)
        self.samples.append((rover_view, state, rover.latent_state.reshape(-1), target))


def rollout(
    udp: morphing_rover_UDP, rover: Rover, experts: dict[int, Expert], horizon: int
) -> tuple[list[tuple[float, float]], list[Sample]]:
    """Drive each scenario of `experts` (0-29) with the rover's current controller.

    Returns the (normalised distance, normalised time) result of every scenario and the expert-labelled samples
    of their first `horizon` steps.
    """
    results, samples = [], []
    with torch.no_grad():
        for scenario, expert in experts.items():
            map_id, scenario_id = divmod(scenario, SCENARIOS_PER_MAP)
            distance, time = udp.run_single_scenario(rover, map_id, scenario_id,
                                                     on_step=_Recorder(expert, horizon, samples))
            results.append((float(distance), float(time)))
    return results, samples


def fit(controller: Controller, samples: list[Sample], n_iter: int, learning_rate: float) -> float:
    """Full-batch regression of the controller's orientation change on the expert's; returns the last loss."""
    views = torch.stack([sample[0] for sample in samples]).unsqueeze(1)
    states = torch.stack([sample[1] for sample in samples])
    latents = torch.stack([sample[2] for sample in samples])
    targets = torch.tensor([sample[3] for sample in samples], dtype=torch.float64)
    optimiser = torch.optim.Adam(controller.parameters(), learning_rate)
    # NNPACK, picked by default for batched CPU convolutions, is ~6x slower than the fallback on the strided
    # 11x11 convolution of the controller (same results).
    with torch.backends.nnpack.flags(enabled=False):
        for _ in range(n_iter):
            _, angular_change, _ = controller(views, states, latents)
            loss = ((MAX_DA * angular_change - targets) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    # Keep the chromosome valid for submission.
    with torch.no_grad():
        for param in controller.parameters():
            param.clamp_(FLOAT_MIN, FLOAT_MAX)
    return loss.item()


def train_controller(
    udp: morphing_rover_UDP, chromosome: np.ndarray, cfg: ImitationConfig, output_dir: Path | None = None
) -> tuple[np.ndarray, float]:
    """Train the controller of `chromosome` by imitation learning (its masks are left untouched).

    Returns the best chromosome met and its fitness, averaged over the training scenarios. With an `output_dir`,
    the best chromosome is saved there whenever it improves, along with a history.csv of the training.

    Each rollout collecting samples is a full simulation, hence also an evaluation of the controller before it is
    fitted to these samples. A last rollout (logged with horizon 0) evaluates the final controller.
    """
    scenarios = range(TOTAL_NUM_SCENARIOS) if cfg.scenarios is None else cfg.scenarios
    experts = {s: make_expert(s, cfg.radius_factors.get(s), cfg.arc_points, cfg.lookahead) for s in scenarios}
    rover = Rover(chromosome)
    rover.Control.requires_grad_(True)
    tracker = _Tracker(chromosome, output_dir)

    buffer: list[Sample] = []
    for run in range(1, cfg.n_runs + 1):
        for horizon in range(cfg.horizon_start, cfg.horizon_stop + 1, cfg.horizon_step):
            results, samples = rollout(udp, rover, experts, horizon)
            tracker.evaluate(rover, results)
            buffer = samples if cfg.buffer_size is None else (buffer + samples)[-cfg.buffer_size:]
            loss = fit(rover.Control, buffer, cfg.n_iter, cfg.learning_rate)
            tracker.log(run, horizon, len(buffer), loss)
    results, _ = rollout(udp, rover, experts, horizon=0)
    tracker.evaluate(rover, results)
    tracker.log(cfg.n_runs, 0, 0, float("nan"))
    return tracker.best_chromosome, tracker.best_fitness


class _Tracker:
    """Keeps the best chromosome met and logs the training progress."""

    def __init__(self, chromosome: np.ndarray, output_dir: Path | None):
        self.best_chromosome = np.asarray(chromosome, dtype=np.float64)
        self.best_fitness = float("inf")
        self.output_dir = output_dir
        if output_dir is not None:
            with open(output_dir / "history.csv", "w", newline="") as f:
                csv.writer(f).writerow(["run", "horizon", "fitness", "reached", "samples", "loss"])

    def evaluate(self, rover: Rover, results: list[tuple[float, float]]) -> None:
        self.fitness = float(np.mean([(1 + distance) * time for distance, time in results]))
        self.reached = f"{sum(distance == 0 for distance, _ in results)}/{len(results)}"
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_chromosome = np.asarray(rover.chromosome, dtype=np.float64)
            if self.output_dir is not None:
                save_chromosome(self.output_dir / "best_chromosome.npy", self.best_chromosome)

    def log(self, run: int, horizon: int, n_samples: int, loss: float) -> None:
        logger.info("run %d horizon %d: fitness %.4f (best %.4f), %s targets reached, %d training samples, "
                    "loss %.5f", run, horizon, self.fitness, self.best_fitness, self.reached, n_samples, loss)
        if self.output_dir is not None:
            with open(self.output_dir / "history.csv", "a", newline="") as f:
                csv.writer(f).writerow([run, horizon, self.fitness, self.reached, n_samples, loss])
