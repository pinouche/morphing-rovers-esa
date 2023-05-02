import pickle
import argparse
import numpy as np
import torch
import os

from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from morphing_rovers.src.neural_network_supervised.morphing_udp_modified import MAX_TIME, morphing_rover_UDP

PATH_CONTROL = "./neural_network_supervised/optimized_control.p"
PATH_MASKS = "./mode_optimization/experiments/optimized_masks.p"

if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    network_trainer = None

    if os.path.exists(PATH_CONTROL):
        control = pickle.load(open(PATH_CONTROL, "rb"))
    else:
        control = morphing_rover_UDP().example()

    if os.path.exists(PATH_MASKS):
        masks_tensors = pickle.load(open(PATH_MASKS, "rb"))
    else:
        masks_tensors = [torch.rand(11, 11, requires_grad=True) for _ in range(4)]

    masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
    chromosome = np.concatenate((masks, control.chromosome))
    chromosome[628] = 10000  # change bias to always switch mode

    for n_iter in range(1, MAX_TIME+1):
        print(f"Optimizing network for the {n_iter} first rover's steps")

        network_trainer = OptimizeNetworkSupervised(options, chromosome)
        network_trainer.train(n_iter)

        print("AVERAGE ROVER'S SPEED: ", np.mean(network_trainer.udp.rover.overall_speed))

        chromosome = np.concatenate((masks, network_trainer.udp.rover.Control.chromosome))  # updated chromosome
        chromosome[628] = 10000

        print(f"Completed {network_trainer.completed_scenarios} scenarios")

    # compute fitness
    fitness = network_trainer.udp.fitness(chromosome)
    print("fitness", fitness, "overall speed", np.mean(network_trainer.udp.rover.overall_speed))

    pickle.dump(network_trainer.udp.rover.Control, open(PATH_CONTROL, "wb"))
