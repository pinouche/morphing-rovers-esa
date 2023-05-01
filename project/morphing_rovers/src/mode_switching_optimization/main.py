import pickle
import argparse
import numpy as np

from utils import load_chromosome
from optimization import OptimizeNetworkSupervisedSwitching
from morphing_udp_modified import MAX_TIME


if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    chromosome = load_chromosome()
    network_trainer = OptimizeNetworkSupervisedSwitching(options, chromosome)
    network_trainer.train(MAX_TIME)


