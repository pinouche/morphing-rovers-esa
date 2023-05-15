import numpy as np
import torch

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from morphing_rovers.morphing_udp import MAX_TIME
from morphing_rovers.src.mode_optimization.utils import velocity_function
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME, Rover


def create_random_chromosome():

    chromosome = morphing_rover_UDP().example()
    rover = Rover(chromosome)
    control = rover.Control
    masks_tensors = [torch.rand(11, 11, requires_grad=True) for _ in range(4)]
    chromosome = update_chromosome_with_mask(masks_tensors, control.chromosome, always_switch=True)

    return masks_tensors, chromosome

def get_best_mode(mode_view, masks_list):

    velocities = []
    for m in masks_list:
        velocity = velocity_function(torch.unsqueeze(m, dim=0), mode_view).numpy(force=True)
        velocities.append(velocity)
    best_mode = np.argmax(velocities)

    return best_mode


def adjust_clusters(cluster_data, masks_tensors):
    views = cluster_data[0]

    clusters_list = []
    for i, v in enumerate(views):
        best_mode = get_best_mode(v, masks_tensors)
        clusters_list.append(best_mode)

    cluster_data[1] = np.array(clusters_list)

    return cluster_data


def init_modes(options, chromosome):

    # initial run to get the dataset for clustering
    network_trainer = OptimizeNetworkSupervised(options, chromosome)
    network_trainer.train(MAX_TIME, train=False)
    path_data = network_trainer.udp.rover.cluster_data

    # clustering
    cluster_trainer = ClusteringTerrain(options, data=path_data, groupby_scenario=True, random_state=0)
    cluster_trainer.run()
    cluster_trainer_output = cluster_trainer.output

    # optimize modes
    mode_trainer = OptimizeMask(options, data=cluster_trainer_output)
    mode_trainer.train()
    average_speed = mode_trainer.weighted_average
    masks_tensors = mode_trainer.optimized_masks

    return masks_tensors, cluster_trainer_output, average_speed


def adjust_clusters_and_modes(options, cluster_trainer_output, masks_tensors, best_average_speed):

    # adjust clusters and optimize masks again
    iteration_number = 0
    early_stopping_counter = 0
    while True:
        cluster_trainer_output = adjust_clusters(cluster_trainer_output, masks_tensors)
        mode_trainer = OptimizeMask(options, data=cluster_trainer_output)
        mode_trainer.train()
        new_average_speed = mode_trainer.weighted_average
        # print(f"The weighted average speed is: {new_average_speed} and the cluster sizes are {np.unique(cluster_trainer_output[1], return_counts=True)}")
        masks_tensors = mode_trainer.optimized_masks

        if new_average_speed > best_average_speed:
            early_stopping_counter += 1
        else:
            best_average_speed = new_average_speed
            early_stopping_counter = 0

        if early_stopping_counter == 5:
            break

        if iteration_number == 10:
            break

        iteration_number += 1

    return masks_tensors, cluster_trainer_output


def update_chromosome_with_mask(masks_tensors, chromosome, always_switch=True):
    # updated chromosome
    masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
    chromosome = np.concatenate((masks, chromosome))

    if always_switch:
        chromosome[628] = 10000
    else:
        chromosome[628] = 0

    return chromosome

def fitness_wrapper(x):

    func = morphing_rover_UDP().fitness
    fitness = round(func(x)[0], 4)
    print(f"the fitness is {fitness}")

    return fitness
