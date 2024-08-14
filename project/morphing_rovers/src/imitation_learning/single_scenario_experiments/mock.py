import numpy as np

from morphing_rovers.src.imitation_learning.single_scenario_experiments.arc_trajectories import get_coordinates, compute_both_arcs


for scenario_n in range(30):
    start, end = get_coordinates(scenario_n)
    dist = np.sqrt(np.sum((end-start)**2))
    if dist > 2500:
        print(scenario_n, dist)
