import pickle
import numpy as np
import argparse
import torch
import os

from morphing_rovers.utils import load_config
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME
from morphing_rovers.src.imitation_learning.optimization import OptimizeNetworkSupervised
from morphing_rovers.src.utils import update_chromosome_with_mask, create_random_chromosome
from morphing_rovers.src.imitation_learning.arc_trajectories import get_coordinates, compute_both_arcs

N_RUNS = 100
PATH_CHROMOSOME = "../trained_chromosomes/chromosome_fitness_2.0211.p"

config = load_config("config.yml")


def func(i):
    torch.manual_seed(i)  # add 10 every time to add randomness

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    scenario_n = config["scenario_number"]

    udp = morphing_rover_UDP()

    if os.path.exists(PATH_CHROMOSOME):
        print("Chromosome exists")
        chromosome = pickle.load(open(PATH_CHROMOSOME, "rb"))
        chromosome[4*11*11:-7] = np.random.randn(len(chromosome[4*11*11:-7]))
        masks_tensors = [
            torch.tensor(np.reshape(chromosome[11 ** 2 * i:11 ** 2 * (i + 1)], (11, 11)), requires_grad=True) for i
            in range(4)]
    else:
        masks_tensors, chromosome = create_random_chromosome()

    # fitness = udp.fitness(chromosome)[0]
    udp.pretty(chromosome)
    # print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

    start, end = get_coordinates(scenario_n)
    dist = np.sqrt(np.sum((end-start)**2))

    # for radius in list(np.arange(dist/2, dist*2, dist/10)) + [1000]:  # 1000 is basically a straight line from to start to end
    for radius in [1000]:
        arcs = compute_both_arcs(start, end, radius)
        best_fitness = np.inf
        for arc in arcs:  # we have the arc clockwise and the arc counter-clockwise
            for i in range(N_RUNS):
                print(f"Running for run number {i}")
                for n_iter in range(MAX_TIME, MAX_TIME + 1):
                    print(f"Optimizing network for the {n_iter} first rover's steps")

                    network_trainer = OptimizeNetworkSupervised(options, chromosome, scenario_n, arc)
                    network_trainer.train(n_iter, train=True)

                    chromosome = update_chromosome_with_mask(masks_tensors,
                                                             network_trainer.udp.rover.Control.chromosome,
                                                             always_switch=True)

                    # fitness = udp.fitness(chromosome)[0]
                    udp.pretty(chromosome)
                    # udp.plot(chromosome)

                    print("FITNESS AFTER PATH LEARNING", np.inf, "overall speed", np.mean(udp.rover.overall_speed),
                          "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))

                    # print("FITNESS AFTER PATH LEARNING", fitness, "overall speed", np.mean(udp.rover.overall_speed),
                    #       "average distance from objectives:", np.mean(network_trainer.udp.rover.overall_distance))
                    #
                    # if fitness < best_fitness:
                    #     print("NEW BEST FITNESS!!")
                    #     if fitness < 2.05:
                    #         pickle.dump(chromosome,
                    #                     open(f"../trained_chromosomes/chromosome_fitness_{round(fitness, 4)}.p", "wb"))
                    #     best_fitness = fitness


if __name__ == "__main__":

    func(0)
