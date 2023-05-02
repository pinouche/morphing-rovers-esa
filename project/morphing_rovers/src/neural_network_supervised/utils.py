import pickle
import torch
import numpy as np

from morphing_rovers.src.mode_optimization.utils import velocity_function


def get_best_mode(mode_view, masks_list):

    velocities = []
    for m in masks_list:
        velocity = velocity_function(torch.unsqueeze(m, dim=0), mode_view).numpy(force=True)
        velocities.append(velocity)
    best_mode = np.argmax(velocities)

    return best_mode


