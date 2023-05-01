import pickle
import numpy as np


def load_chromosome():
    masks = pickle.load(open("../mode_optimization/experiments/optimized_masks.p", "rb"))
    control = pickle.load(open("../neural_network_supervised/optimized_control.p", "rb"))
    masks = np.array([m.numpy(force=True) for m in masks]).flatten()
    chromosome = np.concatenate((masks, control.chromosome))

    return chromosome
