import pickle
import numpy as np
import argparse

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from morphing_rovers.morphing_udp import morphing_rover_UDP, MAX_TIME
from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from utils import init_modes, adjust_clusters_and_modes


if __name__ == "__main__":

    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()
    masks_tensors = pickle.load(open("../mode_optimization/experiments/optimized_masks.p", "rb"))
    control = pickle.load(open("optimized_control.p", "rb"))

    # set-up the chromosome
    masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
    chromosome = np.concatenate((masks, control.chromosome))
    chromosome[628] = 10000  # set switching mode always to be on

    # initial run to get the dataset for clustering
    masks_tensors, cluster_trainer_output, best_average_speed = init_modes(options, chromosome)
    print(f"The weighted average speed is: {best_average_speed} and the cluster sizes are {np.unique(cluster_trainer_output[1], return_counts=True)}")
    pickle.dump(masks_tensors, open("../mode_optimization/experiments/optimized_masks.p", "wb"))

    if len(np.unique(cluster_trainer_output[1])) != 1:
        # adjust clusters and optimize masks again iteratively
        masks_tensors = adjust_clusters_and_modes(options, cluster_trainer_output, masks_tensors, best_average_speed)

    # updated chromosome
    masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
    chromosome = np.concatenate((masks, control.chromosome))
    chromosome[628] = 10000

    fitness = udp.fitness(chromosome)
    print("initial fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

    for i in range(10):
        print(f"COMPUTING FOR ITERATION NUMBER {i}")
        for n_iter in range(1, MAX_TIME+1):
            print(f"Optimizing network for the {n_iter} first rover's steps")

            network_trainer = OptimizeNetworkSupervised(options, chromosome)
            network_trainer.train(n_iter)
            path_data = network_trainer.udp.rover.cluster_data

            print("AVERAGE ROVER'S SPEED: ", np.mean(network_trainer.udp.rover.overall_speed))

            chromosome = np.concatenate((masks, network_trainer.udp.rover.Control.chromosome))  # updated chromosome
            chromosome[628] = 10000

        # compute fitness
        fitness = udp.fitness(chromosome)
        print("round number", i, "fitness", fitness, "overall speed", np.mean(udp.rover.overall_speed))

        # clustering
        cluster_trainer = ClusteringTerrain(options, path_data)
        cluster_trainer.run()
        cluster_trainer_output = cluster_trainer.output

        # optimize modes
        mode_trainer = OptimizeMask(options, masks_tensors, cluster_trainer_output)
        mode_trainer.train()
        best_average_speed = mode_trainer.weighted_average
        masks_tensors = mode_trainer.optimized_masks

        if len(np.unique(cluster_trainer_output[1])) != 1:
            masks_tensors = adjust_clusters_and_modes(options, cluster_trainer_output, masks_tensors, best_average_speed)

        # updated chromosome
        masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
        chromosome = np.concatenate((masks, control.chromosome))
        chromosome[628] = 10000

    pickle.dump(chromosome, open("chromosome.p", "wb"))

    udp.plot(chromosome, plot_modes=True, plot_mode_efficiency=True)
    udp.pretty(chromosome)

