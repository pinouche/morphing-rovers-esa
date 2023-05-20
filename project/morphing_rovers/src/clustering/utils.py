import os
import torch
import numpy as np

from morphing_rovers.src.autoencoder.models.model import Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(session_name: str = "session_1", latent_dim: int = 50, fc_dim: int = 256) -> Autoencoder:

    checkpoint_name = os.path.join(".", "autoencoder", "experiments", session_name)
    checkpoint = torch.load(checkpoint_name, map_location=device)

    model = Autoencoder(latent_dim, fc_dim)
    model.load_state_dict(checkpoint['model'])

    # print('Loaded model and optimiser weights from {}\n'.format(checkpoint_name))

    return model


def swap_most_and_least_occurring_clusters(clusters):
    counts = np.unique(clusters, return_counts=True)
    max_index = counts[0][np.argmax(counts[1])]  # most occurring

    dict_swap = {max_index: 0, 0: max_index}

    new_arr = []

    for v in clusters:
        if v in dict_swap.keys():
            v = dict_swap[v]
        new_arr.append(v)

    return new_arr

