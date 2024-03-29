import pickle
import numpy as np
import argparse
import torch

from morphing_rovers.utils import load_config
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME
from morphing_rovers.src.imitation_learning.full_scenarios_experiments.optimization import OptimizeNetworkSupervised
# from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from morphing_rovers.src.utils import update_chromosome_with_mask, get_chromosome_from_path

N_RUNS = 100
PATH_CHROMOSOME = "../trained_chromosomes/chromosome_fitness_2.0211.p"

config = load_config("./full_scenarios_experiments/config.yml")
SCENARIOS_LIST = list(config["radius_dictionary"].keys())


def func(i):
    torch.manual_seed(i)  # add 10 every time to add randomness

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    masks_tensors, chromosome = get_chromosome_from_path(PATH_CHROMOSOME, True)
    training_data = []
    for i in range(N_RUNS):
        print(f"Running for run number {i}")
        for n_iter in range(1, MAX_TIME + 1):
            print(f"Optimizing network for the {n_iter} first rover's steps")

            network_trainer = OptimizeNetworkSupervised(options, chromosome, training_data)
            network_trainer.train(n_iter, train=True)
            training_data = network_trainer.training_data

            chromosome = update_chromosome_with_mask(masks_tensors,
                                                     network_trainer.udp.rover.Control.chromosome,
                                                     always_switch=True)

            # network_trainer = OptimizeNetworkSupervised(options, chromosome)
            # network_trainer.train(n_iter, train=True)
            # path_data = network_trainer.udp.rover.cluster_data
            #
            # chromosome = update_chromosome_with_mask(masks_tensors,
            #                                          network_trainer.udp.rover.Control.chromosome,
            #                                          always_switch=True)

            score, _ = udp.pretty(chromosome, SCENARIOS_LIST)
            # udp.plot(chromosome, SCENARIOS_LIST)

            print("FITNESS AFTER PATH LEARNING", score[0], "overall speed", np.mean(udp.rover.overall_speed),
                  "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))


if __name__ == "__main__":
    func(0)
