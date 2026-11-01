"""Split, assemble, save and load chromosomes (decision vectors).

A chromosome is [4 masks of 11x11 | network biases | network weights | 7 network hyperparameters].
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.utils.hooks
from torch import nn

from morphing_rovers.udp import (
    FLOAT_MAX,
    FLOAT_MIN,
    MASK_SIZE,
    NETWORK_SETUP,
    NUM_BIASES,
    NUM_MODE_PARAMETERS,
    NUM_NN_PARAMS,
    NUMBER_OF_MODES,
    Controller,
    get_bounds,
)

MASKS_SHAPE = (NUMBER_OF_MODES, MASK_SIZE, MASK_SIZE)
CHROMOSOME_SIZE = NUM_MODE_PARAMETERS + NUM_NN_PARAMS + 7

# The network outputs are [mode switch command, angular change]: their biases are the last two biases
# and their weights (shape [2, hidden]) the last weights of the network.
_HIDDEN = NETWORK_SETUP["hidden_neurons"][1]
SWITCH_BIAS = NUM_MODE_PARAMETERS + NUM_BIASES - 2
SWITCH_WEIGHTS = slice(NUM_MODE_PARAMETERS + NUM_NN_PARAMS - 2 * _HIDDEN, NUM_MODE_PARAMETERS + NUM_NN_PARAMS - _HIDDEN)

# Pooling and activations of a wired controller: ReLU on the convolutions, and hardtanh on the three layers of the
# steering channel, as it is the identity on [-1, 1].
WIRED_GENES = (0, 1, 6, 3, 6, 3, 3)
# Hidden channels of a wired controller kept apart from training, in each of its three hidden layers.
STEERING_CHANNEL = 0
OFFSET_CHANNEL = 1
RESERVED_CHANNELS = [STEERING_CHANNEL, OFFSET_CHANNEL]
# Index in the rover state of the mode efficiency and of the angle to the sample divided by pi.
STATE_EFFICIENCY = 0
STATE_ANGLE_TO_SAMPLE = 2
# Gain from this angle to the steering command: `MAX_DA * 4 * state[STATE_ANGLE_TO_SAMPLE]` is the angle itself.
STEERING_GAIN = 4.0
# Largest heading offset (degrees) a wired controller can encode, and its steering command.
MAX_HEADING_OFFSET = 30.0
OFFSET_GAIN = STEERING_GAIN * MAX_HEADING_OFFSET / 180.0


def split(chromosome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the masks, shaped (4, 11, 11), and the network part of a chromosome."""
    chromosome = np.asarray(chromosome, dtype=np.float64)
    return chromosome[:NUM_MODE_PARAMETERS].reshape(MASKS_SHAPE), chromosome[NUM_MODE_PARAMETERS:]


def assemble(masks: np.ndarray, network: np.ndarray) -> np.ndarray:
    """Concatenate masks and network parameters into a chromosome that lies within the problem bounds."""
    chromosome = np.concatenate([np.asarray(masks, dtype=np.float64).ravel(), np.asarray(network, dtype=np.float64)])
    if chromosome.shape != (CHROMOSOME_SIZE,):
        raise ValueError(f"Chromosome has {chromosome.size} values, expected {CHROMOSOME_SIZE}.")
    return clip_to_bounds(chromosome)


def always_switch(chromosome: np.ndarray) -> np.ndarray:
    """Make the rover switch to the best mode whenever its cooldown allows it.

    The mode switch command becomes a constant positive bias, which keeps the chromosome within bounds
    (unlike the bias of 10000 used previously).
    """
    chromosome = np.array(chromosome, dtype=np.float64)
    chromosome[SWITCH_WEIGHTS] = 0.0
    chromosome[SWITCH_BIAS] = FLOAT_MAX
    return chromosome


def reset_switch_head(chromosome: np.ndarray) -> np.ndarray:
    """Random switch output, undoing `always_switch`, whose saturated bias gradient descent could not bring back."""
    chromosome = np.array(chromosome, dtype=np.float64)
    chromosome[SWITCH_WEIGHTS] = (0.15 * torch.randn(_HIDDEN, dtype=torch.float64)).numpy()
    chromosome[SWITCH_BIAS] = 0.0
    return chromosome


def wire_controller(controller: Controller) -> None:
    """Build the reserved channels of a wired controller, which steers exactly like an expert keeping a constant angle
    from the sample, specific to each scenario, whatever the other channels learn.

    - Steering channel: carries the angle to the sample through the three hidden layers to the steering output.
    - Offset channel, which holds the heading offset of the scenario. The first layer detects the first step, the only
      one with a null mode efficiency (it is at least 1/9 afterwards). The second one lets a code of the scenario
      through on that step only, or -1, which has no effect, afterwards. The last one, the only recurrent channel of
      the network, adds it to its own previous value, so it stores the code for the rest of the scenario.

    The code itself comes from the convolutions and is set by `encode_offsets`. Other channels are free to read these
    two, and drive the switch output.
    """
    with torch.no_grad():
        for param in (controller.inp.weight, controller.lin2.weight, controller.lin3.weight, controller.lin4.weight):
            param[RESERVED_CHANNELS] = 0.0
        for layer, source in ((controller.inp, STATE_ANGLE_TO_SAMPLE), (controller.lin3, STEERING_CHANNEL),
                              (controller.lin4, STEERING_CHANNEL)):
            layer.weight[STEERING_CHANNEL, source] = 1.0
            layer.bias[STEERING_CHANNEL] = 0.0
        controller.inp.weight[OFFSET_CHANNEL, STATE_EFFICIENCY] = FLOAT_MIN
        controller.inp.bias[OFFSET_CHANNEL] = 1.0
        controller.lin3.weight[OFFSET_CHANNEL, OFFSET_CHANNEL] = FLOAT_MAX
        controller.lin3.bias[OFFSET_CHANNEL] = FLOAT_MIN
        controller.lin4.weight[OFFSET_CHANNEL, OFFSET_CHANNEL] = 0.5
        controller.lin4.bias[OFFSET_CHANNEL] = 0.5
        controller.recurr.weight.zero_()
        controller.recurr.weight[OFFSET_CHANNEL, OFFSET_CHANNEL] = 1.0
        controller.output.weight[1] = 0.0
        controller.output.weight[1, STEERING_CHANNEL] = STEERING_GAIN
        controller.output.weight[1, OFFSET_CHANNEL] = 2 * OFFSET_GAIN
        controller.output.bias[1] = -OFFSET_GAIN


def encode_offsets(controller: Controller, first_views: torch.Tensor, offsets: np.ndarray) -> None:
    """Give a wired controller the heading offset (degrees) of each scenario.

    `first_views` (N, 89, 89) are what the rover sees on the first step of each scenario, which tells them apart. The
    weights from the convolutional features to the offset channel are solved in closed form for each view to produce
    its offset, so this must be called again whenever the convolutions change.
    """
    if np.abs(offsets).max(initial=0.0) > MAX_HEADING_OFFSET:
        raise ValueError(f"Heading offsets are limited to {MAX_HEADING_OFFSET} degrees, got {offsets}.")
    with torch.no_grad():
        features = controller.activation3(controller.conv2(controller.pool1(controller.activation1(
            controller.conv(first_views.unsqueeze(1))))))
        features = controller.pool2(features).flatten(1).double()
        codes = torch.as_tensor(offsets / MAX_HEADING_OFFSET, dtype=torch.float64)
        weights = torch.linalg.pinv(features) @ codes
        if weights.abs().max() > FLOAT_MAX:
            raise ValueError(f"Heading offsets need weights up to {weights.abs().max():.1f}, out of the bounds.")
        controller.lin2.weight[OFFSET_CHANNEL] = weights.float()


def freeze_wiring(controller: Controller) -> list[torch.utils.hooks.RemovableHandle]:
    """Keep the reserved channels of a wired controller out of training, until the returned hooks are removed."""
    rows = {controller.recurr.weight: slice(None), controller.output.weight: [1], controller.output.bias: [1]}
    for layer in (controller.inp, controller.lin3, controller.lin4):
        rows[layer.weight] = rows[layer.bias] = RESERVED_CHANNELS
    rows[controller.lin2.weight] = RESERVED_CHANNELS

    def zero_rows(selection):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            gradient = gradient.clone()
            gradient[selection] = 0.0
            return gradient

        return hook

    return [param.register_hook(zero_rows(selection)) for param, selection in rows.items()]


def wired_network() -> np.ndarray:
    """Network part of a chromosome with wired steering (see `wire_controller`) and a random, untrained switch output."""
    controller = Controller(np.concatenate([np.zeros(NUM_NN_PARAMS), WIRED_GENES]))
    for layer in (controller.conv, controller.conv2):
        nn.init.kaiming_uniform_(layer.weight.data, a=2.2)
    for layer in (controller.inp, controller.lin2, controller.lin3, controller.lin4, controller.output):
        layer.weight.data.normal_(0, 0.15)
    wire_controller(controller)
    return np.array(controller.chromosome, dtype=np.float64)


def clip_to_bounds(chromosome: np.ndarray) -> np.ndarray:
    lower, upper = get_bounds()
    return np.clip(chromosome, lower, upper)


def load_chromosome(path: str | Path) -> np.ndarray:
    """Load a chromosome saved with `save_chromosome`, or pickled by the previous version of this code."""
    path = Path(path)
    if path.suffix == ".npy":
        chromosome = np.load(path)
    else:
        with open(path, "rb") as f:
            chromosome = pickle.load(f)
    return np.asarray(chromosome, dtype=np.float64)


def save_chromosome(path: str | Path, chromosome: np.ndarray) -> None:
    np.save(path, np.asarray(chromosome, dtype=np.float64))
