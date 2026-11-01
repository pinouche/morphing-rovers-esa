"""Imitation learning of the rover controller (DAgger-style), for its two outputs.

The rover drives with its current controller while an expert tells, at every visited state, what it should have done.

Steering: the controller is fitted to the expert's orientation change, then the number of recorded steps per scenario
(the horizon) grows, so that the controller learns to correct its own mistakes along the whole trajectory. Keeping
a constant angle from the sample needs none of this with a wired controller (see `chromosome.wire_controller`).

Switching: the rover may decline a morph into the best mode for its current terrain, and should when the terrain
ahead does not justify the cooldown. `routes.SwitchOracle` labels these decisions, weighted by the steps at stake.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from morphing_rovers.chromosome import encode_offsets, freeze_wiring, save_chromosome, split
from morphing_rovers.config import ImitationConfig, SwitchingConfig
from morphing_rovers.experts import Expert, make_expert
from morphing_rovers.routes import Route, SwitchOracle, Trajectory, first_views, schedule
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

    def __call__(self, rover: Rover, rover_view, mode_view, distance_vector, original_distance) -> None:
        if self.steps == self.horizon:
            return
        self.steps += 1
        target = self.expert.steering(rover.position.numpy(), rover.angle)
        state = rover.network_state(distance_vector, original_distance)
        self.samples.append((rover_view, state, rover.latent_state.reshape(-1), target))


def make_experts(cfg: ImitationConfig) -> dict[int, Expert]:
    """Expert of each scenario trained on, heading straight for the sample unless it has a radius factor."""
    scenarios = range(TOTAL_NUM_SCENARIOS) if cfg.scenarios is None else cfg.scenarios
    return {s: make_expert(s, cfg.radius_factors.get(s), cfg.arc_points, cfg.lookahead, cfg.heading_offsets.get(s, 0.0))
            for s in scenarios}


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


def _stack(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    views = torch.stack([sample[0] for sample in samples]).unsqueeze(1)
    states = torch.stack([sample[1] for sample in samples])
    latents = torch.stack([sample[2] for sample in samples])
    targets = torch.tensor([sample[3] for sample in samples], dtype=torch.float64)
    return views, states, latents, targets


def fit(
    controller: Controller, samples: list[Sample], n_iter: int, learning_rate: float, batch_size: int | None = None
) -> float:
    """Regression of the controller's orientation change on the expert's; returns the last loss.

    Each Adam step uses all the samples, or with a `batch_size` a random minibatch, whose cost doesn't depend on
    the size of the dataset.
    """
    full_batch = None if batch_size is not None and batch_size < len(samples) else _stack(samples)
    optimiser = torch.optim.Adam(controller.parameters(), learning_rate)
    # NNPACK, picked by default for batched CPU convolutions, is ~6x slower than the fallback on the strided
    # 11x11 convolution of the controller (same results).
    with torch.backends.nnpack.flags(enabled=False):
        for _ in range(n_iter):
            if full_batch is None:
                indices = torch.randint(len(samples), (batch_size,)).tolist()
                views, states, latents, targets = _stack([samples[i] for i in indices])
            else:
                views, states, latents, targets = full_batch
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
    udp: morphing_rover_UDP,
    chromosome: np.ndarray,
    cfg: ImitationConfig,
    experts: dict[int, Expert] | None = None,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, float]:
    """Train the controller of `chromosome` by imitation learning (its masks are left untouched).

    Returns the best chromosome met and its fitness, averaged over the training scenarios. With an `output_dir`,
    the best chromosome is saved there whenever it improves, along with a history.csv of the training.

    Each rollout collecting samples is a full simulation, hence also an evaluation of the controller before it is
    fitted to these samples. A last rollout (logged with horizon 0) evaluates the final controller.
    """
    experts = make_experts(cfg) if experts is None else experts
    rover = Rover(chromosome)
    rover.Control.requires_grad_(True)
    tracker = _Tracker(chromosome, output_dir, "history.csv", ["run", "horizon", "fitness", "reached", "samples", "loss"])

    buffer: list[Sample] = []
    for run in range(1, cfg.n_runs + 1):
        for horizon in range(cfg.horizon_start, cfg.horizon_stop + 1, cfg.horizon_step):
            results, samples = rollout(udp, rover, experts, horizon)
            tracker.evaluate(rover, results)
            buffer = samples if cfg.buffer_size is None else (buffer + samples)[-cfg.buffer_size:]
            loss = fit(rover.Control, buffer, cfg.n_iter, cfg.learning_rate, cfg.batch_size)
            tracker.log(run, horizon, len(buffer), loss)
    results, _ = rollout(udp, rover, experts, horizon=0)
    tracker.evaluate(rover, results)
    tracker.log(cfg.n_runs, 0, 0, float("nan"))
    return tracker.best_chromosome, tracker.best_fitness


@dataclass
class Decision:
    """A step where the switch output matters: the rover can morph, into another mode than its current one."""

    view: torch.Tensor
    state: torch.Tensor
    latent: torch.Tensor
    scenario: int
    position: np.ndarray
    mode: int
    best: int
    advantage: int = 0  # steps saved by morphing, according to the oracle


class _Driver:
    """Controller of a rover whose switch command the oracle can overrule."""

    def __init__(self, controller: Controller):
        self.controller = controller
        self.forced: bool | None = None

    @property
    def chromosome(self) -> list[float]:
        return self.controller.chromosome

    def __call__(self, landscape: torch.Tensor, state: torch.Tensor, latent: torch.Tensor):
        switch, steering, latent = self.controller(landscape, state, latent)
        if self.forced is not None:
            switch = torch.tensor([1.0 if self.forced else -1.0])
        return switch, steering, latent


class _DecisionRecorder:
    """`on_step` hook recording the decisions of a scenario, its trajectory and, if any, its steering samples."""

    def __init__(self, driver: _Driver, scenario: int, oracle: SwitchOracle, oracle_share: float,
                 steering_expert: Expert | None):
        self.driver = driver
        self.scenario = scenario
        self.oracle = oracle
        self.oracle_share = oracle_share
        self.steering_expert = steering_expert
        self.decisions: list[Decision] = []
        self.samples: list[Sample] = []
        self.trajectory = Trajectory()
        self.result: tuple[float, float] | None = None  # (normalised distance, normalised time)

    def __call__(self, rover: Rover, rover_view, mode_view, distance_vector, original_distance) -> None:
        self.trajectory(rover)
        self.driver.forced = None
        mode = rover.current_mode
        best = mode if rover.cooldown > 0 else int(rover.get_best_mode(mode_view))
        if best == mode and self.steering_expert is None:
            return
        position = rover.position.numpy().astype(np.float64)
        latent = rover.latent_state.reshape(-1)
        state = rover.network_state(distance_vector, original_distance)
        if self.steering_expert is not None:
            self.samples.append((rover_view, state, latent, self.steering_expert.steering(position, rover.angle)))
        if best != mode:
            self.decisions.append(Decision(rover_view, state, latent, self.scenario, position, mode, best))
            if torch.rand(1).item() < self.oracle_share:
                self.driver.forced = self.oracle.advantage(self.oracle.route.arc_length(position), mode, best) > 0


def fit_switching(
    controller: Controller,
    decisions: list[Decision],
    cfg: SwitchingConfig,
    samples: list[Sample] | None = None,
) -> tuple[float, float]:
    """Cost-sensitive classification of the decisions by the switch output; with steering `samples`, also a
    regression of the steering output on them, for a learned controller not to forget how to steer.

    Returns the last loss and the accuracy on the decisions, weighted by the steps at stake. Without `samples`, the
    steering of the controller is wired, and stays so.
    """
    if not decisions:
        return float("nan"), float("nan")
    views = torch.stack([decision.view for decision in decisions]).unsqueeze(1)
    states = torch.stack([decision.state for decision in decisions])
    latents = torch.stack([decision.latent for decision in decisions])
    labels = torch.tensor([decision.advantage > 0 for decision in decisions], dtype=torch.float32) * 2 - 1
    weights = torch.tensor([abs(decision.advantage) for decision in decisions], dtype=torch.float32)
    weights = weights.clamp(max=cfg.max_weight)

    optimiser = torch.optim.Adam(controller.parameters(), cfg.learning_rate)
    wiring = [] if samples else freeze_wiring(controller)
    with torch.backends.nnpack.flags(enabled=False):
        for _ in range(cfg.n_iter):
            batch = torch.randint(len(decisions), (cfg.batch_size,))
            command, _, _ = controller(views[batch], states[batch], latents[batch])
            loss = (weights[batch] * F.softplus(-labels[batch] * command)).sum() / weights[batch].sum()
            if samples:
                indices = torch.randint(len(samples), (cfg.batch_size,)).tolist()
                sample_views, sample_states, sample_latents, targets = _stack([samples[i] for i in indices])
                _, angular_change, _ = controller(sample_views, sample_states, sample_latents)
                loss = loss + cfg.steering_weight * ((MAX_DA * angular_change - targets) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        for hook in wiring:
            hook.remove()
        with torch.no_grad():
            for param in controller.parameters():
                param.clamp_(FLOAT_MIN, FLOAT_MAX)
            commands = torch.cat([controller(view, state, latent)[0] for view, state, latent
                                  in zip(views.split(1024), states.split(1024), latents.split(1024), strict=True)])
    correct = ((commands > 0) == (labels > 0)).float()
    return loss.item(), ((correct * weights).sum() / weights.sum()).item()


def train_switching(
    udp: morphing_rover_UDP,
    chromosome: np.ndarray,
    experts: dict[int, Expert],
    routes: dict[int, Route],
    cfg: SwitchingConfig,
    steering: str = "wired",
    output_dir: Path | None = None,
) -> tuple[np.ndarray, float]:
    """Train the switch output of `chromosome` to imitate the oracle (its masks are left untouched).

    The oracle drives a decreasing share of the decisions of the first rounds, then the controller drives alone, and
    only these rollouts count as evaluations. With `steering="wired"`, the heading offsets of the experts are encoded
    into the network, and again after each fit as they depend on the convolutions. With `steering="learned"`, the
    steering output keeps being fitted to the experts, and as the rover then strays from `routes`, they are rebuilt
    from each rollout to label its decisions. Returns the best chromosome met and its fitness; with an `output_dir`,
    the best chromosome is saved whenever it improves, along with a switching.csv of the training.
    """
    masks = split(chromosome)[0]
    rover = Rover(chromosome)
    controller = rover.Control
    controller.requires_grad_(True)
    driver = rover.Control = _Driver(controller)
    wired = steering == "wired"
    views = first_views(udp.env)
    offsets = np.array([experts[s].heading_offset if s in experts else 0.0 for s in range(TOTAL_NUM_SCENARIOS)])
    if wired:
        encode_offsets(controller, views, offsets)
    tracker = _Tracker(chromosome, output_dir, "switching.csv",
                       ["round", "oracle_share", "fitness", "reached", "decisions", "loss", "accuracy"])
    _, oracles, _ = schedule(masks, routes)
    decisions: list[Decision] = []
    samples: list[Sample] = []
    for round_id in range(1, cfg.rounds + 1):
        share = cfg.oracle_shares[round_id - 1] if round_id <= len(cfg.oracle_shares) else 0.0
        recorders = _drive(udp, rover, driver, experts, oracles, share, record_steering=not wired)
        results = [recorder.result for recorder in recorders]
        if share == 0:
            tracker.evaluate(rover, results)
        if not wired:
            routes = {r.scenario: r.trajectory.route(udp.env, r.scenario, r.result[0]) for r in recorders}
            _, oracles, _ = schedule(masks, routes)
        new_decisions = [decision for recorder in recorders for decision in recorder.decisions]
        decisions += _labelled(new_decisions, oracles)
        samples = (samples + [s for r in recorders for s in r.samples])[-cfg.steering_buffer:]
        loss, accuracy = fit_switching(controller, decisions, cfg, samples)
        if wired:
            encode_offsets(controller, views, offsets)
        fitness, reached = _summary(results)
        logger.info("round %d, oracle share %.2f: fitness %.4f (best %.4f), %s targets reached, %d decisions, "
                    "loss %.4f, accuracy %.4f", round_id, share, fitness, tracker.best_fitness, reached,
                    len(decisions), loss, accuracy)
        tracker.record(round_id, share, fitness, reached, len(decisions), loss, accuracy)
    recorders = _drive(udp, rover, driver, experts, oracles, 0.0, record_steering=False)
    tracker.evaluate(rover, [recorder.result for recorder in recorders])
    logger.info("final rollout: fitness %.4f (best %.4f), %s targets reached", tracker.fitness, tracker.best_fitness,
                tracker.reached)
    tracker.record(cfg.rounds, 0.0, tracker.fitness, tracker.reached, 0, float("nan"), float("nan"))
    return tracker.best_chromosome, tracker.best_fitness


def _drive(
    udp: morphing_rover_UDP,
    rover: Rover,
    driver: _Driver,
    experts: dict[int, Expert],
    oracles: dict[int, SwitchOracle],
    oracle_share: float,
    record_steering: bool,
) -> list[_DecisionRecorder]:
    """Simulate each scenario with the rover's controller, returning the recorders of their steps and results."""
    recorders = []
    with torch.no_grad():
        for scenario, expert in experts.items():
            recorder = _DecisionRecorder(driver, scenario, oracles[scenario], oracle_share,
                                         expert if record_steering else None)
            map_id, scenario_id = divmod(scenario, SCENARIOS_PER_MAP)
            distance, time = udp.run_single_scenario(rover, map_id, scenario_id, on_step=recorder)
            recorder.result = (float(distance), float(time))
            recorders.append(recorder)
    return recorders


def _summary(results: list[tuple[float, float]]) -> tuple[float, str]:
    """Mean fitness and number of targets reached."""
    fitness = float(np.mean([(1 + distance) * time for distance, time in results]))
    return fitness, f"{sum(distance == 0 for distance, _ in results)}/{len(results)}"


def _labelled(decisions: list[Decision], oracles: dict[int, SwitchOracle]) -> list[Decision]:
    """Decisions labelled by the oracle, without the ties, which do not matter."""
    for decision in decisions:
        oracle = oracles[decision.scenario]
        arc_length = oracle.route.arc_length(decision.position)
        decision.advantage = oracle.advantage(arc_length, decision.mode, decision.best)
    return [decision for decision in decisions if decision.advantage != 0]


class _Tracker:
    """Keeps the best chromosome met and logs the training progress."""

    def __init__(self, chromosome: np.ndarray, output_dir: Path | None, history: str, columns: list[str]):
        self.best_chromosome = np.asarray(chromosome, dtype=np.float64)
        self.best_fitness = float("inf")
        self.output_dir = output_dir
        self.history = None if output_dir is None else output_dir / history
        if self.history is not None:
            with open(self.history, "w", newline="") as f:
                csv.writer(f).writerow(columns)

    def evaluate(self, rover: Rover, results: list[tuple[float, float]]) -> None:
        self.fitness, self.reached = _summary(results)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_chromosome = np.asarray(rover.chromosome, dtype=np.float64)
            if self.output_dir is not None:
                save_chromosome(self.output_dir / "best_chromosome.npy", self.best_chromosome)

    def log(self, run: int, horizon: int, n_samples: int, loss: float) -> None:
        logger.info("run %d horizon %d: fitness %.4f (best %.4f), %s targets reached, %d training samples, "
                    "loss %.5f", run, horizon, self.fitness, self.best_fitness, self.reached, n_samples, loss)
        self.record(run, horizon, self.fitness, self.reached, n_samples, loss)

    def record(self, *row) -> None:
        if self.history is not None:
            with open(self.history, "a", newline="") as f:
                csv.writer(f).writerow(row)
