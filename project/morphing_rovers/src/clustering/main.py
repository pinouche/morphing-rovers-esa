import argparse
from morphing_rovers.src.clustering.clustering_model.clustering import ClusteringTerrain


if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    cluster_trainer = ClusteringTerrain(options)
    cluster_trainer.run()

