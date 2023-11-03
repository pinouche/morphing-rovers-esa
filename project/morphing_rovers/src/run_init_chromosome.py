import copy
import pickle
import numpy as np
import argparse
import torch
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import random

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME, Rover
from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from utils import create_random_chromosome, compute_average_best_velocity
from morphing_rovers.src.utils import get_chromosome_from_path, update_chromosome_with_mask

PATH_CHROMOSOME = "./trained_chromosomes/chromosome_fitness_2.0211.p"
SCENARIOS_LIST = list(range(30))
N_ITERATIONS_FULL_RUN = 200
N_STEPS_TO_RUN = 100
CLUSTERBY_SCENARIO = True


def func(i):
    torch.manual_seed(i)  # add 10 every time to add randomness

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    masks_tensors, chromosome = get_chromosome_from_path(PATH_CHROMOSOME, True)

    fitness = udp.fitness(chromosome, SCENARIOS_LIST)[0]
    print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

    best_fitness = np.inf
    for j in range(N_ITERATIONS_FULL_RUN):
        print(f"COMPUTING FOR RUN NUMBER {j}")
        for n_iter in range(1, MAX_TIME + 1):
            print(f"Optimizing network for the {n_iter} first rover's steps")

            network_trainer = OptimizeNetworkSupervised(options, chromosome)
            network_trainer.train(n_iter, train=True)
            path_data = network_trainer.udp.rover.cluster_data

            chromosome = update_chromosome_with_mask(masks_tensors,
                                                     network_trainer.udp.rover.Control.chromosome,
                                                     always_switch=True)

            fitness = udp.fitness(chromosome, SCENARIOS_LIST)[0]
            udp.pretty(chromosome, SCENARIOS_LIST)
            udp.plot(chromosome, SCENARIOS_LIST)

            print("FITNESS AFTER PATH LEARNING", fitness, "overall speed", np.mean(udp.rover.overall_speed),
                  "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))

            if fitness < best_fitness:
                print("NEW BEST FITNESS!!")
                if fitness < 2.05:
                    pickle.dump(chromosome,
                                open(f"./trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                best_fitness = fitness

            # clustering
            # cluster_trainer = ClusteringTerrain(options, path_data=path_data, groupby_scenario=CLUSTERBY_SCENARIO,
            #                                     random_state=j)
            # cluster_trainer.run()
            # cluster_trainer_output = cluster_trainer.output
            # scenarios_id = cluster_trainer.scenarios_id
            #
            # if CLUSTERBY_SCENARIO:
            #     c = [cluster_trainer_output[1], cluster_trainer_output[-1]]
            #     dict_replace = dict(zip(np.unique(scenarios_id), c[-1]))
            #     clusters = np.array([dict_replace[k] for k in scenarios_id])
            #     c = [cluster_trainer_output[0], clusters]
            # else:
            #     c = [cluster_trainer_output[0], cluster_trainer_output[-1]]
            #
            # # optimize modes
            # mode_trainer = OptimizeMask(options, data=c)
            # mode_trainer.train()
            # masks_tensors = mode_trainer.optimized_masks
            #
            # velocity = compute_average_best_velocity(cluster_trainer_output[1], masks_tensors)
            #
            # # updated chromosome
            # chromosome = update_chromosome_with_mask(masks_tensors,
            #                                          network_trainer.udp.rover.Control.chromosome,
            #                                          always_switch=True)


if __name__ == "__main__":

    func(0)

    # seeds = [0, 1, 2, 3]
    # with ThreadPoolExecutor() as executor:
    #     batch_list = executor.map(func, seeds)
