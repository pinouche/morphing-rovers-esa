import copy
import pickle
import numpy as np
import argparse
import torch
import os
from loguru import logger

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME, Rover
from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from utils import init_modes, adjust_clusters_and_modes, update_chromosome_with_mask

PATH_CHROMOSOME = "./trained_chromosomes/chromosome_iteration_1.p"
N_ITERATIONS_FULL_RUN = 1


if __name__ == "__main__":

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    if os.path.exists(PATH_CHROMOSOME):
        chromosome = pickle.load(open(PATH_CHROMOSOME, "rb"))
    else:
        chromosome = morphing_rover_UDP().example()
        rover = Rover(chromosome)
        control = rover.Control
        masks_tensors = [torch.rand(11, 11, requires_grad=True) for _ in range(4)]
        chromosome = update_chromosome_with_mask(masks_tensors, control.chromosome, always_switch=True)

    fitness = udp.fitness(chromosome)
    # udp.plot(chromosome)
    print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))
    # initial run to get the dataset for clustering
    masks_tensors, cluster_trainer_output, best_average_speed = init_modes(options, chromosome)

    # if len(np.unique(cluster_trainer_output[1])) != 1:
    #     # adjust clusters and optimize masks again iteratively
    #     masks_tensors = adjust_clusters_and_modes(options, cluster_trainer_output, masks_tensors, best_average_speed)

    # updated chromosome
    masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
    chromosome[:11*11*4] = masks
    fitness = udp.fitness(chromosome)[0]
    print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

    for j in range(1):
        for n_iter in range(1, MAX_TIME+1):
            if n_iter % 1 == 0:
                print(f"Optimizing network for the {n_iter} first rover's steps")

                network_trainer = OptimizeNetworkSupervised(options, chromosome)
                network_trainer.train(n_iter, train=True)
                path_data = network_trainer.udp.rover.cluster_data

                chromosome = update_chromosome_with_mask(masks_tensors,
                                                         network_trainer.udp.rover.Control.chromosome,
                                                         always_switch=True)

                fitness = udp.fitness(chromosome)[0]
                print("FITNESS AFTER PATH LEARNING", fitness, "overall speed", np.mean(udp.rover.overall_speed),
                      "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))

                # clustering
                cluster_trainer = ClusteringTerrain(options, data=path_data, groupby_scenario=True, random_state=j)
                cluster_trainer.run()
                cluster_trainer_output = cluster_trainer.output

                # optimize modes
                mode_trainer = OptimizeMask(options, data=cluster_trainer_output)
                mode_trainer.train()
                best_average_speed = mode_trainer.weighted_average
                masks_tensors = mode_trainer.optimized_masks

                # if len(np.unique(cluster_trainer_output[1])) != 1:
                #     masks_tensors = adjust_clusters_and_modes(options, cluster_trainer_output, masks_tensors,
                #                                               best_average_speed)

                # updated chromosome
                chromosome = update_chromosome_with_mask(masks_tensors,
                                                        network_trainer.udp.rover.Control.chromosome,
                                                         always_switch=True)

                # compute fitness
                fitness = udp.fitness(chromosome)[0]
                print("FITNESS AFTER MODE OPTIMIZATION", fitness, "overall speed", np.mean(udp.rover.overall_speed))


    pickle.dump(chromosome, open(f"./trained_chromosomes/chromosome_iteration_{i}.p", "wb"))

    udp.plot(chromosome, plot_modes=True, plot_mode_efficiency=True)
    udp.pretty(chromosome)

