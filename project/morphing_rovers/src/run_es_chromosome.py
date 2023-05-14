import pickle
import argparse
import os

from morphing_rovers.morphing_udp import morphing_rover_UDP
from morphing_rovers.src.evolution_strategies.evolution_strategies import EvolutionStrategies

PATH_CHROMOSOME = "./trained_chromosomes/chromosome_fitness_2.0268.p"


if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    # load pre-trained chromosome
    if os.path.exists(PATH_CHROMOSOME):
        chromosome = pickle.load(open(PATH_CHROMOSOME, "rb"))
    else:
        raise FileNotFoundError

    es_trainer = EvolutionStrategies(options, chromosome)
    es_trainer.fit()


