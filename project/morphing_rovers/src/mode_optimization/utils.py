import pickle
import torch
from morphing_rovers.morphing_udp import EPS_C


def load_data(path="../clustering/experiments/clusters.p"):
    data = pickle.load(open(path, "rb"))
    return data


def velocity_function(form, mode_view):
    '''
    Velocity function that maps a rover form and current terrain to a velocity.
    Composed of a term taking into account shape only (distance)
    as well as height difference scale only (luminance).
    Args:
        form: mask of the rover mode
        mode_view: terrain height map the rover is standing on.
    Returns:
        Scalar between 0 and 1 that scales the velocity.
        Rover velocity is obtained by multiplying this factor with MAX_VELOCITY.
    '''
    # calculate norm of vectors
    f_norm = form.norm(dim=(1, 2))
    mv_norm = mode_view.norm(dim=(1, 2))
    # luminance = how well do the vector norms agree?
    # 0 = no match, 1 = perfect match
    luminance = (2 * f_norm * mv_norm + EPS_C) / (f_norm ** 2 + mv_norm ** 2 + EPS_C)
    f_norm = torch.reshape(f_norm.repeat_interleave(121), (f_norm.shape[0], 11, 11))
    mv_norm = torch.reshape(mv_norm.repeat_interleave(121), (mv_norm.shape[0], 11, 11))
    # distance = Euclidean distance on unit sphere (i.e., of normalized vectors)
    # Rescaled to go from 0 to 1, with 0 = no match, 1 = perfect match
    distance = (2 - (form / f_norm - mode_view / mv_norm).norm(dim=(1, 2))) * 0.5
    # final metric = product of distance and luminance
    metric = distance * luminance.sqrt()
    # turn metric into range 0 to 2
    # 0 = perfect match, 2 = worst match
    metric = (1 - metric) * 2

    return distance_to_velocity(metric)


def distance_to_velocity(x):
    '''Helper function that turns the initial score for rover and terrain similarity into a velocity.'''
    return 1. / (1 + x ** 3)
