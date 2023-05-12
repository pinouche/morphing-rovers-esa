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
from utils import init_modes, adjust_clusters_and_modes, update_chromosome_with_mask, create_random_chromosome

PATH_CHROMOSOME = "./trained_chromosomes/chromosome_iteration_1.p"
N_ITERATIONS_FULL_RUN = 20
N_STEPS_TO_RUN = 100
CLUSTERBY_SCENARIO = True


if __name__ == "__main__":

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    if os.path.exists(PATH_CHROMOSOME):
        chromosome = pickle.load(open(PATH_CHROMOSOME, "rb"))
        # initial run to get the dataset for clustering
        masks_tensors, cluster_trainer_output, best_average_speed = init_modes(options, chromosome)
        # updated chromosome
        masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
        chromosome[:11*11*4] = masks
        fitness = udp.fitness(chromosome)[0]
        print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))
    else:
        masks_tensors, chromosome = create_random_chromosome()

    fitness = udp.fitness(chromosome)
    print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))
    #initial run to get the dataset for clustering
    # masks_tensors, cluster_trainer_output, best_average_speed = init_modes(options, chromosome)

    # if len(np.unique(cluster_trainer_output[1])) != 1:
    #     # adjust clusters and optimize masks again iteratively
    #     masks_tensors = adjust_clusters_and_modes(options, cluster_trainer_output, masks_tensors, best_average_speed)

    # updated chromosome
    # masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
    # chromosome[:11*11*4] = masks
    # fitness = udp.fitness(chromosome)[0]
    # print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

    best_fitness = np.inf
    fitness_list = [[] for _ in range(N_ITERATIONS_FULL_RUN)]
    for j in range(N_ITERATIONS_FULL_RUN):

        # get a new randomly generated chromosome to start a new round of optimization
        if j > 0:
            masks_tensors, chromosome = create_random_chromosome()

        for n_iter in range(1, N_STEPS_TO_RUN+1):
            if n_iter % 1 == 0:
                print(f"Optimizing network for the {n_iter} first rover's steps")

                network_trainer = OptimizeNetworkSupervised(options, chromosome)
                network_trainer.train(n_iter, train=True)
                path_data = network_trainer.udp.rover.cluster_data

                chromosome = update_chromosome_with_mask(masks_tensors,
                                                         network_trainer.udp.rover.Control.chromosome,
                                                         always_switch=True)

                fitness = udp.fitness(chromosome)[0]
                fitness_list[j].append(fitness)
                print("FITNESS AFTER PATH LEARNING", fitness, "overall speed", np.mean(udp.rover.overall_speed),
                      "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))

                if fitness < best_fitness:
                    print("NEW BEST FITNESS!!")
                    pickle.dump(chromosome, open(f"./trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                    best_fitness = fitness

                # clustering
                cluster_trainer = ClusteringTerrain(options, data=path_data, groupby_scenario=CLUSTERBY_SCENARIO,
                                                    random_state=j+100)
                cluster_trainer.run()
                cluster_trainer_output = cluster_trainer.output

                # optimize modes
                mode_trainer = OptimizeMask(options, data=cluster_trainer_output)
                mode_trainer.train()
                best_average_speed = mode_trainer.weighted_average
                masks_tensors = mode_trainer.optimized_masks

                # if len(np.unique(cluster_trainer_output[1])) != 1:
                #     masks_tensors, cluster_trainer_output = adjust_clusters_and_modes(options, cluster_trainer_output,
                #                                                                       masks_tensors,
                #                                                                       best_average_speed)
                #
                #     if CLUSTERBY_SCENARIO:
                #         scenarios = np.arange(0, 30, 1)
                #         dict_replace = dict(zip(scenarios, cluster_trainer_output[-1]))
                #         cluster_trainer_output[-1] = np.array([dict_replace[k] for k in cluster_trainer.scenarios_id])

                # updated chromosome
                chromosome = update_chromosome_with_mask(masks_tensors,
                                                        network_trainer.udp.rover.Control.chromosome,
                                                         always_switch=True)

                # compute fitness
                fitness = udp.fitness(chromosome)[0]
                fitness_list[j].append(fitness)
                print("FITNESS AFTER MODE OPTIMIZATION", fitness, "overall speed", np.mean(udp.rover.overall_speed))

                if fitness < best_fitness:
                    print("NEW BEST FITNESS!!")
                    pickle.dump(chromosome, open(f"./trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                    best_fitness = fitness

                pickle.dump(fitness_list, open(f"./trained_chromosomes/fitness_list.p", "wb"))

    udp.plot(chromosome, plot_modes=True, plot_mode_efficiency=True)
    udp.pretty(chromosome)

