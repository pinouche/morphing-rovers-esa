import copy
import pickle
import numpy as np
import argparse
import torch
import os
import multiprocessing
import random
from loguru import logger

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME, Rover
from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from utils import adjust_clusters_and_modes, update_chromosome_with_mask, create_random_chromosome

PATH_CHROMOSOME = "./trained_chromosomes/chromosome_fitness_fine_tuned_does_not_exist.p"
N_ITERATIONS_FULL_RUN = 10
N_STEPS_TO_RUN = 200
CLUSTERBY_SCENARIO = True


def func(i, return_dict):

    torch.manual_seed(i+10) # add 10 every time to add randomness

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    return_dict[i] = []
    best_fitness = np.inf
    fitness_list = [[] for _ in range(N_ITERATIONS_FULL_RUN)]
    for j in range(N_ITERATIONS_FULL_RUN):
        print(f"COMPUTING FOR RUN NUMBER {j}")

        if os.path.exists(PATH_CHROMOSOME):
            chromosome = pickle.load(open(PATH_CHROMOSOME, "rb"))
            # initial run to get the dataset for clustering
            masks_tensors = [
                torch.tensor(np.reshape(chromosome[11 ** 2 * i:11 ** 2 * (i + 1)], (11, 11)), requires_grad=True) for i
                in range(4)]
            # masks_tensors, cluster_trainer_output, best_average_speed = init_modes(options, chromosome)
        else:
            masks_tensors, chromosome = create_random_chromosome()

        fitness = udp.fitness(chromosome)[0]
        print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

        for n_iter in range(1, N_STEPS_TO_RUN + 1):
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
                    pickle.dump(chromosome,
                                open(f"./trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                    best_fitness = fitness
                    return_dict[i].append((chromosome, fitness))

                # clustering
                cluster_trainer = ClusteringTerrain(options, data=path_data, groupby_scenario=CLUSTERBY_SCENARIO,
                                                    random_state=j)
                cluster_trainer.run()
                cluster_trainer_output = cluster_trainer.output
                scenarios_id = cluster_trainer.scenarios_id

                if CLUSTERBY_SCENARIO:
                    c = [cluster_trainer_output[1], cluster_trainer_output[-1]]
                else:
                    c = [cluster_trainer_output[0], cluster_trainer_output[-1]]

                # optimize modes
                mode_trainer = OptimizeMask(options, data=c)
                mode_trainer.train()
                best_average_speed = mode_trainer.weighted_average
                masks_tensors = mode_trainer.optimized_masks

                if CLUSTERBY_SCENARIO:
                    # here, we want to adjust the scenarios' average
                    ################################################### remove this to use the average only
                    masks_tensors, c = adjust_clusters_and_modes(options, c, masks_tensors, best_average_speed)
                    # scenarios = np.arange(0, 30, 1)
                    # print("C[-1]", c[-1], "scenarios_id", scenarios_id)
                    dict_replace = dict(zip(np.unique(scenarios_id), c[-1]))
                    # print("dict_replace", dict_replace)
                    clusters = np.array([dict_replace[k] for k in scenarios_id])
                    # print("clusters", clusters)
                    c[-1] = clusters
                    c[0] = cluster_trainer_output[0]
                    #########################################

                    mode_trainer = OptimizeMask(options, data=c)
                    mode_trainer.train()
                    best_average_speed = mode_trainer.weighted_average
                    masks_tensors = mode_trainer.optimized_masks

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
                    pickle.dump(chromosome,
                                open(f"./trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                    best_fitness = fitness
                    return_dict[i].append((chromosome, fitness))

                pickle.dump(fitness_list, open(f"./trained_chromosomes/fitness_list.p", "wb"))

    # udp.plot(chromosome, plot_modes=True, plot_mode_efficiency=True)
    # udp.pretty(chromosome)


if __name__ == "__main__":

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores

    p = [multiprocessing.Process(target=func, args=(i, return_dict))
         for i in range(2)]

    for proc in p:
        proc.start()
    for proc in p:
        proc.join()

    individuals_fitness = return_dict.values()

    print(individuals_fitness)
