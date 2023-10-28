import pickle
import numpy as np
import argparse
import torch
import os

from morphing_rovers.utils import load_config
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME
from morphing_rovers.src.imitation_learning.single_scenario_experiments.optimization import OptimizeNetworkSupervised
from morphing_rovers.src.utils import update_chromosome_with_mask, get_chromosome_from_path
from morphing_rovers.src.imitation_learning.single_scenario_experiments.arc_trajectories import get_coordinates, compute_both_arcs

N_RUNS = 50
PATH_CHROMOSOME = "./../trained_chromosomes/chromosome_fitness_2.0211.p"

config = load_config("./full_scenarios_experiments/config.yml")


def func(i):
    torch.manual_seed(i)

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    for scenario_n in [2, 3, 4, 5, 8, 12, 24, 25, 27, 29]:
        start, end = get_coordinates(scenario_n)
        dist = np.sqrt(np.sum((end-start)**2))
        radius = dist
        for factor in np.arange(1, 10, 1):
            radius *= factor
            print(f"Computing for factor {factor} and radius {radius}.")
            dic_result = dict()
            arc_num = 0
            arcs = compute_both_arcs(start, end, radius)
            masks_tensors, chromosome = get_chromosome_from_path(PATH_CHROMOSOME)
            for arc in [arcs[0]]:  # we have the arc clockwise and the arc counter-clockwise
                dic_result[f"scenario_{scenario_n}_arc_{arc_num}_radius_{np.round(factor, 2)}"] = []
                training_data = []
                for i in range(N_RUNS):
                    print(f"Running for run number {i}")
                    for n_iter in range(MAX_TIME, MAX_TIME + 1):
                        print(f"Optimizing network for the {n_iter} first rover's steps")

                        network_trainer = OptimizeNetworkSupervised(options, chromosome, scenario_n, arc, training_data)
                        network_trainer.train(n_iter, train=True)
                        training_data = network_trainer.training_data

                        chromosome = update_chromosome_with_mask(masks_tensors,
                                                                 network_trainer.udp.rover.Control.chromosome,
                                                                 always_switch=True)

                        score, _ = udp.pretty(chromosome, scenario_n)
                        # udp.plot(chromosome, scenario_n)

                        print("FITNESS AFTER PATH LEARNING", score[0], "overall speed", np.mean(udp.rover.overall_speed),
                              "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))

                        dic_result[f"scenario_{scenario_n}_arc_{arc_num}_radius_{np.round(factor, 2)}"].append(score[0])

            folder_path = f"./single_scenario_experiments/results/scenario_{scenario_n}"
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            pickle.dump(dic_result, open(f"{folder_path}/scenario_{scenario_n}_arc_{arc_num}_radius_{np.round(factor, 2)}.p", "wb"))


if __name__ == "__main__":

    func(0)
