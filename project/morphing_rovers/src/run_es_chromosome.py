import pickle
import argparse
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from utils import create_random_chromosome
from morphing_rovers.morphing_udp import morphing_rover_UDP
from morphing_rovers.src.evolution_strategies.evolution_strategies import EvolutionStrategies

PATH_CHROMOSOME = "./trained_chromosomes/chromosome_fitness_fine_tuned2.0152716636657715.p"
NUM_ITERATIONS = 100


def wrapper_function(seed, opt, chrom):
    es_trainer = EvolutionStrategies(seed, opt, chrom)
    es_trainer.fit()
    return es_trainer.best_chromosome


if __name__ == "__main__":
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    udp = morphing_rover_UDP()

    # load pre-trained chromosome
    if os.path.exists(PATH_CHROMOSOME):
        chromosome = pickle.load(open(PATH_CHROMOSOME, "rb"))
    else:
        _, chromosome = create_random_chromosome()

    seeds = list(range(4))
    with ThreadPoolExecutor() as executor:
        for _ in range(NUM_ITERATIONS):
            results = list(executor.map(wrapper_function, seeds, [options] * len(seeds), [chromosome] * len(seeds)))
            arg_min = np.argmin([udp.fitness(x) for x in results])
            chromosome = np.array(results)[arg_min]




