import pickle
import argparse
import os

from morphing_rovers.morphing_udp import morphing_rover_UDP
from morphing_rovers.src.evolution_strategies.evolution_strategies import EvolutionStrategies

PATH_CHROMOSOME = "../trained_chromosomes/chromosome_iteration_0.p"


if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    # load pre-trained chromosome
    if os.path.exists("./trained_chromosomes/chromosome_iteration_0.p"):
        chromosome = pickle.load(open("./trained_chromosomes/chromosome_iteration_0.p", "rb"))
    else:
        raise FileNotFoundError

    chromosome[628] = 0

    # udp = morphing_rover_UDP()
    # chromosome = udp.example()

    es_trainer = EvolutionStrategies(options, chromosome)
    es_trainer.fit()


