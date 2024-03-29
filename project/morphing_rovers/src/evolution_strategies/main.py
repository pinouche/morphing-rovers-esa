import argparse
import os
import pickle

from morphing_rovers.src.evolution_strategies.evolution_strategies import EvolutionStrategies

if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    # load pre-trained chromosome
    if os.path.exists("../trained_chromosomes/chromosome_iteration_0.p"):
        chromosome = pickle.load(open("../trained_chromosomes/chromosome_iteration_0.p", "rb"))
    else:
        raise FileNotFoundError

    es_trainer = EvolutionStrategies(options, chromosome)
    es_trainer.fit()
