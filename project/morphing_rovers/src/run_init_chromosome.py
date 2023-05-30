import copy
import pickle
import numpy as np
import argparse
import torch
import os
import multiprocessing
import random

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME, Rover
from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from utils import update_chromosome_with_mask, create_random_chromosome

PATH_CHROMOSOME = "./trained_chromosomes/chromosome_fitness_does_not_exist.p"
N_ITERATIONS_FULL_RUN = 20
N_STEPS_TO_RUN = 100
CLUSTERBY_SCENARIO = True


def func(i, return_dict=None):
    torch.manual_seed(i + 600)  # add 10 every time to add randomness

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    return_dict[i] = []
    best_fitness = np.inf
    clusters_list = []
    cluster_trainer_output = None
    for j in range(N_ITERATIONS_FULL_RUN):
        print(f"COMPUTING FOR RUN NUMBER {j}")

        if os.path.exists(PATH_CHROMOSOME):
            chromosome = pickle.load(open(PATH_CHROMOSOME, "rb"))
            masks_tensors = [
                torch.tensor(np.reshape(chromosome[11 ** 2 * i:11 ** 2 * (i + 1)], (11, 11)), requires_grad=True) for i
                in range(4)]
        else:
            masks_tensors, chromosome = create_random_chromosome()

        fitness = udp.fitness(chromosome)[0]
        # udp.pretty(chromosome)
        # udp.plot(chromosome)

        print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

        for n_iter in range(1, MAX_TIME + 1):
            print(f"Optimizing network for the {n_iter} first rover's steps")

            network_trainer = OptimizeNetworkSupervised(options, chromosome)
            network_trainer.train(500, train=True)
            path_data = network_trainer.udp.rover.cluster_data

            chromosome = update_chromosome_with_mask(masks_tensors,
                                                     network_trainer.udp.rover.Control.chromosome,
                                                     always_switch=True)

            fitness = udp.fitness(chromosome)[0]
            print("FITNESS AFTER PATH LEARNING", fitness, "overall speed", np.mean(udp.rover.overall_speed),
                  "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))

            if fitness < best_fitness:
                print("NEW BEST FITNESS!!")
                if fitness < 2.05:
                    pickle.dump(chromosome,
                                open(f"./trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                best_fitness = fitness
                return_dict[i].append((chromosome, fitness))

                if cluster_trainer_output is not None:
                    clusters_list.append((cluster_trainer_output[-1], fitness))
                    pickle.dump(clusters_list, open(f"./trained_chromosomes/clusters_list.p", "wb"))

            # clustering
            cluster_trainer = ClusteringTerrain(options, data=path_data, groupby_scenario=CLUSTERBY_SCENARIO,
                                                random_state=j)
            cluster_trainer.run()
            cluster_trainer_output = cluster_trainer.output
            scenarios_id = cluster_trainer.scenarios_id

            if CLUSTERBY_SCENARIO:
                c = [cluster_trainer_output[1], cluster_trainer_output[-1]]
                dict_replace = dict(zip(np.unique(scenarios_id), c[-1]))
                clusters = np.array([dict_replace[k] for k in scenarios_id])
                c = [cluster_trainer_output[0], clusters]
            else:
                c = [cluster_trainer_output[0], cluster_trainer_output[-1]]

            # optimize modes
            mode_trainer = OptimizeMask(options, data=c)
            mode_trainer.train()
            masks_tensors = mode_trainer.optimized_masks

            # updated chromosome
            chromosome = update_chromosome_with_mask(masks_tensors,
                                                     network_trainer.udp.rover.Control.chromosome,
                                                     always_switch=True)

            # compute fitness
            fitness = udp.fitness(chromosome)[0]
            # udp.pretty(chromosome)
            # udp.plot(chromosome)
            print("FITNESS AFTER MODE OPTIMIZATION", fitness, "overall speed", np.mean(udp.rover.overall_speed))

            if fitness < best_fitness:
                print("NEW BEST FITNESS!!")
                if fitness < 2.05:
                    pickle.dump(chromosome,
                                open(f"./trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                best_fitness = fitness
                return_dict[i].append((chromosome, fitness))

                clusters_list.append((cluster_trainer_output[-1], fitness))
                pickle.dump(clusters_list, open(f"./trained_chromosomes/clusters_list.p", "wb"))

    # udp.plot(chromosome, plot_modes=True, plot_mode_efficiency=True)
    # udp.pretty(chromosome)


if __name__ == "__main__":

    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # func(0, return_dict)

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
