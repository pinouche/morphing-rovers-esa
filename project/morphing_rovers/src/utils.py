import numpy as np
import torch

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
