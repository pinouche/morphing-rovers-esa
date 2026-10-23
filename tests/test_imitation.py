import csv

import numpy as np
import pytest
import torch

from morphing_rovers.chromosome import SWITCH_BIAS, SWITCH_WEIGHTS, always_switch, load_chromosome
from morphing_rovers.config import ImitationConfig
from morphing_rovers.experts import make_expert
from morphing_rovers.imitation import fit, rollout, train_controller
from morphing_rovers.udp import FLOAT_MAX, MAX_DA, Rover, get_bounds


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


def test_train_controller_returns_the_best_chromosome(udp, artifacts, tmp_path):
    chromosome = always_switch(load_chromosome(artifacts / "chromosomes" / "chromosome_2.0469.npy"))
    cfg = ImitationConfig(scenarios=[19, 21], n_runs=2, horizon_start=1, horizon_stop=3, horizon_step=2, n_iter=5,
                          radius_factors={21: 2.0}, buffer_size=7)
    best, best_fitness = train_controller(udp, chromosome, cfg, tmp_path)

    assert best_fitness == pytest.approx(udp.fitness(best, [19, 21])[0], abs=1e-5)
    np.testing.assert_array_equal(load_chromosome(tmp_path / "best_chromosome.npy"), best)
    assert best[SWITCH_BIAS] == FLOAT_MAX and np.all(best[SWITCH_WEIGHTS] == 0)
    with open(tmp_path / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["run"], r["horizon"]) for r in rows] == [("1", "1"), ("1", "3"), ("2", "1"), ("2", "3"), ("2", "0")]
    assert [int(r["samples"]) for r in rows] == [2, 7, 7, 7, 0]
    assert best_fitness == pytest.approx(min(float(r["fitness"]) for r in rows))
