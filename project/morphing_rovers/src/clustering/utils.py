import os
import torch
import random
import numpy as np

from sklearn.decomposition import PCA

from morphing_rovers.src.autoencoder.models.model import Autoencoder
from morphing_rovers.src.mode_optimization.utils import velocity_function

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(session_name: str = "session_1", latent_dim: int = 50, fc_dim: int = 256) -> Autoencoder:

    checkpoint_name = os.path.join(".", "autoencoder", "experiments", session_name)
    checkpoint = torch.load(checkpoint_name, map_location=device)

    model = Autoencoder(latent_dim, fc_dim)
    model.load_state_dict(checkpoint['model'])

    # print('Loaded model and optimiser weights from {}\n'.format(checkpoint_name))

    return model


def swap_values(integers):
    unique_integers, counts = np.unique(integers, return_counts=True)
    # Create a list of tuples with integers and their counts
    integer_counts = list(zip(unique_integers, counts))

    # Sort the list of tuples based on counts in descending order
    sorted_counts = sorted(integer_counts, key=lambda x: x[1], reverse=True)
    keys = dict(zip([tup[0] for tup in sorted_counts], np.arange(len(integers))))

    new_clusters = []
    for val in integers:
        new_integer = keys[val]
        new_clusters.append(new_integer)

    return new_clusters


def compute_velocity_matrix(data):
    V = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            v = velocity_function(data[i], data[j])
            V[i, j] = v

    V = V + V.T - np.diag(np.diag(V))

    return 1-V


def compute_full_velocity_matrix(data, n_landmarks=300):

    n_landmarks = min(data.shape[0], n_landmarks)
    indices = random.sample(range(data.shape[0]), n_landmarks)
    V = np.zeros((data.shape[0], n_landmarks))

    for i in range(data.shape[0]):
        for index, j in enumerate(indices):
            v = velocity_function(data[i], data[j])
            V[i, index] = v

    pca_model = PCA(n_components=20)
    V = pca_model.fit_transform(V)

    return V

