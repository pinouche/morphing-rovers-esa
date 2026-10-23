"""The challenge code must keep producing the fitness values of the official evaluator."""

import numpy as np
import pytest
import torch

from morphing_rovers.chromosome import CHROMOSOME_SIZE, load_chromosome
from morphing_rovers.udp import (
    CENTRE_MIN,
    MASK_CENTRES,
    NUM_BIASES,
    NUM_WEIGHTS,
    SCENARIO_POSITIONS,
    Record,
    Rover,
    get_bounds,
    velocity_function,
)

# Per-scenario fitness computed with the original code (NumPy 1.26).
LEGACY_SCENARIO_FITNESS = {
    "solution": {0: 2.088061, 16: 3.053908, 19: 1.729497, 29: 2.336026},
    "example": {0: 4.86532, 14: 18.061377, 19: 36.645378},
}


def test_problem_dimensions(udp):
    lower, upper = get_bounds()
    assert (NUM_BIASES, NUM_WEIGHTS) == (146, 18496)
    assert len(lower) == len(upper) == len(udp.example()) == CHROMOSOME_SIZE == 19133
    assert all(lower[c] == CENTRE_MIN for c in MASK_CENTRES)
    assert udp.get_bounds() == (lower, upper)


def test_velocity_function():
    view = torch.rand(11, 11) + 0.5
    assert velocity_function(view, view).item() == pytest.approx(1.0)
    assert 0 < velocity_function(torch.randn(11, 11), view).item() < 1


def test_rover_chromosome_round_trip(solution):
    np.testing.assert_allclose(Rover(solution).chromosome, solution, rtol=1e-6)


@pytest.mark.parametrize(("name", "scenario"), [(n, s) for n, d in LEGACY_SCENARIO_FITNESS.items() for s in d])
def test_scenario_fitness_matches_legacy(udp, solution, name, scenario):
    chromosome = solution if name == "solution" else udp.example()
    assert udp.fitness(chromosome, [scenario])[0] == pytest.approx(LEGACY_SCENARIO_FITNESS[name][scenario], abs=1e-5)


def test_fitness_of_subset_is_mean_of_scenarios(udp, solution):
    expected = np.mean([LEGACY_SCENARIO_FITNESS["solution"][s] for s in (0, 19)])
    assert udp.fitness(solution, [0, 19])[0] == pytest.approx(expected, abs=1e-5)


def test_scenarios_are_independent_of_previous_runs(udp, solution):
    rover = Rover(solution)
    first = udp.run_single_scenario(rover, 3, 4)
    udp.run_single_scenario(rover, 0, 0)
    assert udp.run_single_scenario(rover, 3, 4) == first
    assert SCENARIO_POSITIONS[3][4].tolist() == pytest.approx([2380.0, 2123.6, 2598.4, 2006.0])


def test_on_step_hook_sees_every_step_before_the_update(udp, solution):
    seen, record = [], Record()
    rover = Rover(solution)
    result = udp.run_single_scenario(rover, 0, 1, record, on_step=lambda r, *_: seen.append(r.position.tolist()))
    assert result == udp.run_single_scenario(Rover(solution), 0, 1)
    trajectory = np.stack([record[0][1]["x"], record[0][1]["y"]], axis=1)
    np.testing.assert_array_equal(seen, trajectory[:-1])


def test_pretty_marks_skipped_scenarios(udp, solution, capsys):
    (fitness,), record = udp.pretty(solution, [0])
    table = capsys.readouterr().out
    assert fitness == pytest.approx(LEGACY_SCENARIO_FITNESS["solution"][0], abs=1e-5)
    assert record[0][0]["fitness"] and not record[0][1]["fitness"]
    assert "2.088" in table and " - " in table


@pytest.mark.slow
@pytest.mark.parametrize(("name", "fitness"), [("chromosome_2.0211.npy", 2.0211047), ("chromosome_2.0469.npy", 2.0469164)])
def test_full_fitness_of_trained_chromosomes(udp, artifacts, name, fitness):
    assert udp.fitness(load_chromosome(artifacts / "chromosomes" / name))[0] == pytest.approx(fitness, abs=1e-5)
