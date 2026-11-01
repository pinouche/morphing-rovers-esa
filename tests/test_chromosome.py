import pickle

import numpy as np
import pytest
import torch

from morphing_rovers.chromosome import (
    CHROMOSOME_SIZE,
    MASKS_SHAPE,
    WIRED_GENES,
    always_switch,
    assemble,
    clip_to_bounds,
    encode_offsets,
    freeze_wiring,
    load_chromosome,
    reset_switch_head,
    save_chromosome,
    split,
    wired_network,
)
from morphing_rovers.udp import FLOAT_MAX, MAX_DA, NUM_NN_PARAMS, Controller, Rover, get_bounds, minimal_angle_diff


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


def test_wired_network_is_valid_and_steers_exactly_like_the_expert(solution):
    network = wired_network()
    assert network.size == NUM_NN_PARAMS + 7 and tuple(network[-7:]) == WIRED_GENES
    chromosome = assemble(split(solution)[0], network)
    lower, upper = get_bounds()
    assert np.all(chromosome >= lower) and np.all(chromosome <= upper)

    controller = Rover(chromosome).Control
    for angle in np.linspace(-3.1, 3.1, 25):
        state = torch.rand(9)
        state[0], state[2] = 0.0, angle / np.pi  # null efficiency: first step of a scenario
        _, command, _ = controller(torch.randn(89, 89), state, torch.zeros(40))
        expected = np.clip(minimal_angle_diff(angle, 0.0), -MAX_DA, MAX_DA)
        assert MAX_DA * command.item() == pytest.approx(expected, abs=1e-5)


def test_offset_is_set_on_the_first_step_and_held_afterwards():
    views = torch.randn(30, 89, 89) * 0.2
    offsets = np.linspace(-29.0, 29.0, 30)
    controller = Controller(wired_network())
    encode_offsets(controller, views, offsets)
    for scenario in (0, 7, 29):
        state, latent = torch.zeros(9), torch.zeros(40)  # null efficiency: first step
        for step in range(5):
            _, command, latent = controller(views[scenario] if step == 0 else torch.randn(89, 89), state, latent)
            assert np.rad2deg(MAX_DA * command.item()) == pytest.approx(offsets[scenario], abs=1e-3)
            latent = latent[0]
            state = torch.rand(9) * 0.9 + 0.1
            state[2] = 0.0


def test_offsets_beyond_the_range_are_refused():
    with pytest.raises(ValueError, match="limited"):
        encode_offsets(Controller(wired_network()), torch.randn(2, 89, 89), np.array([31.0, 0.0]))


def test_null_offsets_are_a_straight_line():
    controller = Controller(wired_network())
    encode_offsets(controller, torch.randn(30, 89, 89) * 0.2, np.zeros(30))
    state = torch.zeros(9)
    state[2] = 0.1
    _, command, _ = controller(torch.randn(89, 89), state, torch.zeros(40))
    assert MAX_DA * command.item() == pytest.approx(0.1 * np.pi, abs=1e-5)


def test_freeze_wiring_only_keeps_the_reserved_channels_out_of_training():
    controller = Controller(wired_network())
    controller.requires_grad_(True)
    before = np.array(controller.chromosome)
    weights = [param.clone() for param in controller.parameters()]
    hooks = freeze_wiring(controller)
    optimiser = torch.optim.SGD(controller.parameters(), 0.1)
    switch, steering, latent = controller(torch.randn(3, 1, 89, 89) * 0.2, torch.rand(3, 9), torch.rand(3, 40))
    (switch.sum() + steering.sum() + latent.sum()).backward()
    optimiser.step()
    for hook in hooks:
        hook.remove()
    assert not np.array_equal(np.array(controller.chromosome), before)
    assert torch.equal(controller.output.weight[1], weights_of(controller.output, weights, controller)[1])
    assert torch.equal(controller.recurr.weight, weights_of(controller.recurr, weights, controller))
    assert torch.equal(controller.lin3.weight[:2], weights_of(controller.lin3, weights, controller)[:2])
    assert not torch.equal(controller.lin3.weight[2:], weights_of(controller.lin3, weights, controller)[2:])


def weights_of(layer, weights, controller):
    return weights[[id(p) for p in controller.parameters()].index(id(layer.weight))]


def test_reset_switch_head_undoes_always_switch(udp):
    chromosome = reset_switch_head(always_switch(udp.example()))
    controller = Rover(chromosome).Control
    assert controller.output.bias[0] == 0 and torch.any(controller.output.weight[0] != 0)
    np.testing.assert_array_equal(clip_to_bounds(chromosome), chromosome)


def test_save_and_load(tmp_path, solution):
    save_chromosome(tmp_path / "x.npy", solution)
    with open(tmp_path / "x.p", "wb") as f:
        pickle.dump(list(solution), f)
    for path in (tmp_path / "x.npy", tmp_path / "x.p"):
        chromosome = load_chromosome(path)
        assert chromosome.dtype == np.float64 and chromosome.shape == (CHROMOSOME_SIZE,)
        np.testing.assert_array_equal(chromosome, solution)
