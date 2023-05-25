import os
import imageio
import torch
import pickle
import numpy as np
import pandas as pd

from morphing_rovers.morphing_udp import MysteriousMars, MAX_TIME

MAPS_PATH = "../../data/Maps/"
COORDINATES_PATH = "../../data/coordinates.txt"
PATH_CHROMOSOME = "../trained_chromosomes/chromosome_fitness_fine_tuned2.003.p"


def get_map_sizes() -> list:
    map_sizes = []

    for f in os.listdir(MAPS_PATH):
        map_path = MAPS_PATH + "/" + f
        heightmaps = torch.Tensor(imageio.imread_v2(map_path))

        map_sizes.append((heightmaps.shape[0], heightmaps.shape[1]))

    return map_sizes


def load_coordinates():

    data = pd.read_csv(COORDINATES_PATH, sep="\t", header=None)

    return data


def compute_angle_to_sample(sample_position, rover_position):
    distance_vector = sample_position-rover_position
    angle_to_sample = np.arctan2(distance_vector[0], distance_vector[1])
    return angle_to_sample


def create_mode_views_dataset(options, n_samples: int = 250, val_size: float = 0.2) -> None:

    map_size = get_map_sizes()
    coordinates_data = load_coordinates()
    mars = MysteriousMars()

    mode_view_dataset = []
    for map_id in range(len(map_size)):
        subset_coordinates = np.array(coordinates_data[coordinates_data[0] == map_id])

        for landing_id in range(5):
            start_pt = subset_coordinates[landing_id, 1:3]
            end_pt = subset_coordinates[landing_id, 3:]
            for t in np.arange(0, 1+1/n_samples, 1/n_samples):
                new_point = (1-t)*start_pt+t*end_pt
                # here, we set the rover angle to be the same as angle_to_sample (in other words, we go straight to
                # the sample)
                rover_angle = compute_angle_to_sample(end_pt, new_point)
                rover_view, mode_view = mars.extract_local_view(new_point, rover_angle, map_id)
                mode_view_dataset.append(mode_view.numpy())

    mode_view_dataset = np.array(mode_view_dataset)
    np.random.shuffle(mode_view_dataset)
    n = mode_view_dataset.shape[0]
    train_data, val_data = mode_view_dataset[int(val_size*n):], mode_view_dataset[:int(val_size*n)]

    pickle.dump(train_data, open("./training_dataset/train_mode_view_dataset.p", "wb"))
    pickle.dump(val_data, open("./training_dataset/val_mode_view_dataset.p", "wb"))

