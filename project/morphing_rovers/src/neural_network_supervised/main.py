import pickle
import argparse
import numpy as np
import torch

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from optimization import OptimizeNetworkSupervised
from utils import adjust_clusters
from morphing_udp_modified import MAX_TIME, morphing_rover_UDP, MASK_SIZE, NUMBER_OF_MODES


if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    control = pickle.load(open("optimized_control.p", "rb"))
    masks_tensors = pickle.load(open("../mode_optimization/experiments/optimized_masks.p", "rb"))
    # masks_tensors = [torch.rand(11, 11, requires_grad=True) for _ in range(4)]

    masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
    chromosome = np.concatenate((masks, control.chromosome))
    chromosome[628] = 10000  # change bias to always switch mode

    for n_iter in range(1, MAX_TIME+1):
        print(f"Optimizing network for the {n_iter} first rover's steps")

        network_trainer = OptimizeNetworkSupervised(options, chromosome)
        network_trainer.train(n_iter)
        cluster_data = network_trainer.udp.rover.cluster_data

        print("AVERAGE ROVER'S SPEED: ", np.mean(network_trainer.udp.rover.overall_speed))

        # cluster_trainer = ClusteringTerrain(options, cluster_data)
        # cluster_trainer.run()
        # cluster_trainer_output = cluster_trainer.output
        #
        # mode_trainer = OptimizeMask(options, masks_tensors, cluster_trainer_output)
        # mode_trainer.train()
        # masks_tensors = mode_trainer.optimized_masks
        # # updated chromosome
        # masks = np.array([m.numpy(force=True) for m in masks_tensors]).flatten()
        chromosome = np.concatenate((masks, network_trainer.udp.rover.Control.chromosome))  # updated chromosome
        chromosome[628] = 10000

        print(f"Completed {network_trainer.completed_scenarios} scenarios")

    # pickle.dump(network_trainer.udp.rover.Control, open("optimized_control.p", "wb"))
