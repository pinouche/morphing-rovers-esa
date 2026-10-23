import pickle

import numpy as np
import pytest
import torch

from morphing_rovers.chromosome import (
    CHROMOSOME_SIZE,
    MASKS_SHAPE,
    always_switch,
    assemble,
    clip_to_bounds,
    load_chromosome,
    save_chromosome,
    split,
)
from morphing_rovers.udp import FLOAT_MAX, Rover, get_bounds


def test_split_and_assemble_round_trip(solution):
    masks, network = split(solution)
    assert masks.shape == MASKS_SHAPE
    np.testing.assert_array_equal(assemble(masks, network), solution)


def test_assemble_clips_to_bounds_and_checks_size(solution):
    masks, network = split(solution)
    chromosome = assemble(masks * 1e4, network)
    lower, upper = get_bounds()
    assert np.all(chromosome >= lower) and np.all(chromosome <= upper)
    with pytest.raises(ValueError, match="expected 19133"):
        assemble(masks, network[:-1])


def test_always_switch_sets_the_mode_switch_output(udp):
    controller = Rover(always_switch(udp.example())).Control
    assert torch.all(controller.output.weight[0] == 0)
    assert controller.output.bias[0] == FLOAT_MAX
    mode_command, _, _ = controller(torch.randn(5, 1, 89, 89), torch.randn(5, 9), torch.randn(5, 40))
    assert torch.all(mode_command > 0)
    np.testing.assert_array_equal(clip_to_bounds(always_switch(udp.example())), always_switch(udp.example()))


def test_always_switch_keeps_the_fitness_of_a_switching_rover(udp, artifacts):
    # This chromosome switched with a bias of 10000, out of the [-100, 100] bounds.
    chromosome = load_chromosome(artifacts / "chromosomes" / "chromosome_2.0469.npy")
    assert udp.fitness(always_switch(chromosome), [0, 19])[0] == pytest.approx((2.066461 + 1.064306) / 2, abs=1e-5)


def test_save_and_load(tmp_path, solution):
    save_chromosome(tmp_path / "x.npy", solution)
    with open(tmp_path / "x.p", "wb") as f:
        pickle.dump(list(solution), f)
    for path in (tmp_path / "x.npy", tmp_path / "x.p"):
        chromosome = load_chromosome(path)
        assert chromosome.dtype == np.float64 and chromosome.shape == (CHROMOSOME_SIZE,)
        np.testing.assert_array_equal(chromosome, solution)
