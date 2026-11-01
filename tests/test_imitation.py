import csv

import numpy as np
import pytest
import torch

from morphing_rovers.chromosome import (
    SWITCH_BIAS,
    SWITCH_WEIGHTS,
    always_switch,
    assemble,
    encode_offsets,
    load_chromosome,
    reset_switch_head,
    split,
    wired_network,
)
from morphing_rovers.config import ImitationConfig, SwitchingConfig
from morphing_rovers.experts import make_expert
from morphing_rovers.imitation import (
    Decision,
    fit,
    fit_switching,
    make_experts,
    rollout,
    train_controller,
    train_switching,
)
from morphing_rovers.routes import Trajectory, first_views, trace_routes
from morphing_rovers.udp import FLOAT_MAX, MAX_DA, Controller, Rover, get_bounds


@pytest.fixture
def rover(udp) -> Rover:
    rover = Rover(always_switch(udp.example()))
    rover.Control.requires_grad_(True)
    return rover


def test_rollout_records_the_first_steps_and_evaluates(udp, rover):
    experts = {s: make_expert(s, None, 200, 5) for s in (0, 7)}
    results, samples = rollout(udp, rover, experts, horizon=3)
    assert len(samples) == 6
    view, state, latent, target = samples[0]
    assert (view.shape, state.shape, latent.shape) == ((89, 89), (9,), (40,))
    assert isinstance(target, float)
    assert not latent.requires_grad
    assert all(abs(sample[3]) <= MAX_DA for sample in samples)
    fitness = np.mean([(1 + d) * t for d, t in results])
    assert fitness == pytest.approx(udp.fitness(always_switch(udp.example()), [0, 7])[0], abs=1e-5)


def test_fit_learns_the_expert_steering_within_bounds(udp, rover):
    _, samples = rollout(udp, rover, {s: make_expert(s, None, 200, 5) for s in range(0, 30, 3)}, horizon=5)
    switch = (rover.Control.output.weight[0].clone(), rover.Control.output.bias[0].clone())
    first_loss = fit(rover.Control, samples, n_iter=1, learning_rate=0.002)
    assert fit(rover.Control, samples, n_iter=30, learning_rate=0.002) < first_loss
    chromosome = np.asarray(rover.chromosome)
    lower, upper = get_bounds()
    assert np.all(chromosome >= lower) and np.all(chromosome <= upper)
    assert torch.equal(rover.Control.output.weight[0], switch[0]) and rover.Control.output.bias[0] == switch[1]


def test_fit_learns_from_minibatches(udp, rover):
    _, samples = rollout(udp, rover, {s: make_expert(s, None, 200, 5) for s in range(0, 30, 3)}, horizon=5)
    first_loss = fit(rover.Control, samples, n_iter=1, learning_rate=0.002)
    assert fit(rover.Control, samples, n_iter=60, learning_rate=0.002, batch_size=len(samples) // 2) < first_loss


def test_fit_with_a_batch_larger_than_the_dataset_uses_all_the_samples(udp, rover):
    _, samples = rollout(udp, rover, {0: make_expert(0, None, 200, 5)}, horizon=2)
    assert fit(rover.Control, samples, n_iter=1, learning_rate=0.0, batch_size=10) == pytest.approx(
        fit(rover.Control, samples, n_iter=1, learning_rate=0.0))


def test_train_controller_returns_the_best_chromosome(udp, artifacts, tmp_path):
    chromosome = always_switch(load_chromosome(artifacts / "chromosomes" / "chromosome_2.0469.npy"))
    cfg = ImitationConfig(scenarios=[19, 21], n_runs=2, horizon_start=1, horizon_stop=3, horizon_step=2, n_iter=5,
                          radius_factors={21: 2.0}, buffer_size=7)
    best, best_fitness = train_controller(udp, chromosome, cfg, output_dir=tmp_path)

    assert best_fitness == pytest.approx(udp.fitness(best, [19, 21])[0], abs=1e-5)
    np.testing.assert_array_equal(load_chromosome(tmp_path / "best_chromosome.npy"), best)
    assert best[SWITCH_BIAS] == FLOAT_MAX and np.all(best[SWITCH_WEIGHTS] == 0)
    with open(tmp_path / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["run"], r["horizon"]) for r in rows] == [("1", "1"), ("1", "3"), ("2", "1"), ("2", "3"), ("2", "0")]
    assert [int(r["samples"]) for r in rows] == [2, 7, 7, 7, 0]
    assert best_fitness == pytest.approx(min(float(r["fitness"]) for r in rows))


def test_make_experts_follows_the_config():
    experts = make_experts(ImitationConfig(scenarios=[3, 5], radius_factors={5: -2.0}))
    assert list(experts) == [3, 5]
    assert len(experts[3].waypoints) == 1 and len(experts[5].waypoints) == 200
    assert len(make_experts(ImitationConfig())) == 30
    assert make_experts(ImitationConfig(scenarios=[3], heading_offsets={3: 7.0}))[3].heading_offset == 7.0


def decisions_on_terrain(n: int) -> list[Decision]:
    """Decisions whose right answer can be read from the terrain: morph on the high ground only."""
    terrain = torch.randn(n, 89, 89) * 0.01 + torch.arange(n).remainder(2)[:, None, None]
    return [Decision(view, torch.zeros(9), torch.zeros(40), 0, np.zeros(2), 0, 1, 3 if i % 2 else -3)
            for i, view in enumerate(terrain)]


def test_fit_switching_learns_from_the_terrain_and_keeps_the_steering_wired():
    controller = Controller(wired_network())
    controller.requires_grad_(True)
    steering = (controller.output.weight[1].clone(), controller.lin3.weight[0].clone())
    cfg = SwitchingConfig(n_iter=150, batch_size=16)
    _, accuracy = fit_switching(controller, decisions_on_terrain(40), cfg)
    assert accuracy == 1.0
    assert torch.equal(controller.output.weight[1], steering[0]) and torch.equal(controller.lin3.weight[0], steering[1])


def test_fit_switching_without_decisions():
    assert np.isnan(fit_switching(Controller(wired_network()), [], SwitchingConfig())[0])


@pytest.fixture(scope="module")
def short_routes(udp, solution):
    experts = make_experts(ImitationConfig(scenarios=[16, 19]))
    return experts, trace_routes(udp, split(solution)[0], experts)


def test_train_switching_returns_the_best_chromosome(udp, solution, short_routes, tmp_path):
    experts, routes = short_routes
    chromosome = assemble(split(solution)[0], wired_network())
    cfg = SwitchingConfig(rounds=3, oracle_shares=[1.0], n_iter=5, batch_size=8)
    best, best_fitness = train_switching(udp, chromosome, experts, routes, cfg, output_dir=tmp_path)

    assert best_fitness == pytest.approx(udp.fitness(best, [16, 19])[0], abs=1e-5)
    np.testing.assert_array_equal(load_chromosome(tmp_path / "best_chromosome.npy"), best)
    np.testing.assert_array_equal(best[:484], chromosome[:484].astype(np.float32))
    with open(tmp_path / "switching.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["oracle_share"] for r in rows] == ["1.0", "0.0", "0.0", "0.0"]
    # The rollouts driven by the oracle are not evaluations of the controller.
    assert best_fitness == pytest.approx(min(float(r["fitness"]) for r in rows[1:]))


def test_a_wired_network_drives_the_routes_of_experts_with_offsets(udp, solution):
    """The network alone, through the official simulator, against the expert it encodes."""
    masks = split(solution)[0]
    experts = make_experts(ImitationConfig(scenarios=[16, 19], heading_offsets={16: 9.0, 19: -7.0}))
    routes = trace_routes(udp, masks, experts)
    rover = Rover(assemble(masks, wired_network()))
    encode_offsets(rover.Control, first_views(udp.env), np.array([9.0 if s == 16 else -7.0 for s in range(30)]))
    for scenario, route in routes.items():
        trajectory = Trajectory()
        udp.run_single_scenario(rover, *divmod(scenario, 5), on_step=trajectory)
        off_route = [np.linalg.norm(route.positions - p, axis=1).min() for p in trajectory.positions]
        # Two rovers with different velocities turn at different places on the first steps only.
        assert np.mean(off_route) < 3.0 and max(off_route) < 25.0


def test_train_switching_encodes_the_offsets_and_keeps_them_after_training(udp, solution):
    masks = split(solution)[0]
    experts = make_experts(ImitationConfig(scenarios=[16, 19], heading_offsets={16: 9.0, 19: -7.0}))
    routes = trace_routes(udp, masks, experts)
    cfg = SwitchingConfig(rounds=2, oracle_shares=[1.0], n_iter=5, batch_size=8)
    best, best_fitness = train_switching(udp, assemble(masks, wired_network()), experts, routes, cfg)
    assert best_fitness == pytest.approx(udp.fitness(best, [16, 19])[0], abs=1e-5)
    rover = Rover(best)
    trajectory = Trajectory()
    udp.run_single_scenario(rover, 3, 4, on_step=trajectory)
    off_route = [np.linalg.norm(routes[19].positions - p, axis=1).min() for p in trajectory.positions]
    assert np.mean(off_route) < 3.0


def test_train_switching_with_learned_steering_keeps_fitting_the_steering(udp, solution, short_routes):
    experts, routes = short_routes
    chromosome = reset_switch_head(always_switch(solution))
    cfg = SwitchingConfig(rounds=2, oracle_shares=[1.0], n_iter=3, batch_size=8)
    best, best_fitness = train_switching(udp, chromosome, experts, routes, cfg, "learned")
    assert best_fitness == pytest.approx(udp.fitness(best, [16, 19])[0], abs=1e-5)
    assert np.any(split(best)[1] != split(chromosome)[1].astype(np.float32))
