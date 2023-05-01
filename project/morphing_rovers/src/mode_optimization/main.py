import argparse
from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask


if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    cluster_trainer = OptimizeMask(options)
    cluster_trainer.train()
