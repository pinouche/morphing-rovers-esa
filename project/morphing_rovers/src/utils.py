from argparse import Namespace

from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
from morphing_rovers.src.neural_network_supervised.optimization import OptimizeNetworkSupervised
from morphing_rovers.morphing_udp import MAX_TIME

class Config(Namespace):
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)


def adjust_clusters(cluster_data, masks_tensors):
    views = cluster_data[0]

    clusters_list = []
    for i, v in enumerate(views):
        best_mode = get_best_mode(v, masks_tensors)
        clusters_list.append(best_mode)

    cluster_data[1] = np.array(clusters_list)

    return cluster_data


def init_modes(options, chromosome):

    # initial run to get the dataset for clustering
    network_trainer = OptimizeNetworkSupervised(options, chromosome)
    network_trainer.train(MAX_TIME, train=False)
    path_data = network_trainer.udp.rover.cluster_data

    # clustering
    cluster_trainer = ClusteringTerrain(options, path_data)
    cluster_trainer.run()
    cluster_trainer_output = cluster_trainer.output

    # optimize modes
    mode_trainer = OptimizeMask(options, data=cluster_trainer_output)
    mode_trainer.train()
    average_speed = mode_trainer.weighted_average
    masks_tensors = mode_trainer.optimized_masks

    return masks_tensors, cluster_trainer_output, average_speed


def adjust_clusters_and_modes(options, cluster_trainer_output, masks_tensors, best_average_speed):

    # adjust clusters and optimize masks again
    early_stopping_counter = 0
    while True:
        cluster_trainer_output = adjust_clusters(cluster_trainer_output, masks_tensors)
        mode_trainer = OptimizeMask(options, data=cluster_trainer_output)
        mode_trainer.train()
        new_average_speed = mode_trainer.weighted_average
        print(f"The weighted average speed is: {new_average_speed} and the cluster sizes are {np.unique(cluster_trainer_output[1], return_counts=True)}")
        masks_tensors = mode_trainer.optimized_masks

        if new_average_speed > best_average_speed:
            early_stopping_counter += 1
        else:
            best_average_speed = new_average_speed
            early_stopping_counter = 0

        if early_stopping_counter == 5:
            break

    return masks_tensors
