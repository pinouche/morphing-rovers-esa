import numpy as np
import copy
import yaml
import random
import pickle

from morphing_rovers.src.evolution_strategies.utils import get_noise, perturb_chromosome, compute_fitness
from morphing_rovers.morphing_udp import morphing_rover_UDP
from morphing_rovers.utils import Config


# 19126
N_PARAM_TO_PERTURB = 1000


class EvolutionStrategies:

    def __init__(self, options, chromosome):
        self.options = options
        self.chromosome = chromosome
        self.udp = morphing_rover_UDP()
        self.best_fitness = np.inf
        self.best_chromosome = self.chromosome

        ##########
        # Initialise/restore
        ##########
        self.config = None
        config_path = self.options.config
        # Load config file, save it to the experiment output path, and convert to a Config class.
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)

        # init the hyperparameters
        self.sigma = self.config.es_sigma
        self.lr = self.config.es_lr
        self.pop_size = self.config.es_pop_size
        self.epochs = self.config.es_epochs

    def update(self) -> None:

        self.best_fitness = compute_fitness(self.chromosome, self.udp)[0]
        self.udp.pretty(self.chromosome)
        print(f"The current fitness is {self.best_fitness}")

        list_noise = []
        list_fitness = []

        # random_indices = random.sample(range(len(self.chromosome)), N_PARAM_TO_PERTURB)
        for p in range(self.pop_size):
            random_indices = random.sample(range(19126), N_PARAM_TO_PERTURB)
            temporary_chromosome = copy.deepcopy(self.chromosome)

            if p % 10 == 0:
                print(f"Computing for individual number {p}")

            # get the noise
            noise = get_noise(len(random_indices))

            chromosome_to_perturb = copy.deepcopy(temporary_chromosome[random_indices])
            chromosome_to_perturb = perturb_chromosome(chromosome_to_perturb, noise, self.sigma)
            temporary_chromosome[random_indices] = chromosome_to_perturb

            # compute the fitness
            f_obj = round(compute_fitness(temporary_chromosome, self.udp)[0], 4)
            print(f_obj)
            list_fitness.append(f_obj)
            list_noise.append(noise)

            if f_obj < self.best_fitness:
                print(f"new best fitness is {f_obj}")
                pickle.dump(temporary_chromosome, open(f"./trained_chromosomes/chromosome_fitness_fine_tuned{f_obj}.p", "wb"))
                self.best_chromosome = temporary_chromosome
                self.best_fitness = f_obj

        self.chromosome = self.best_chromosome

        # list_weighted_noise = np.array([list_fitness[i]*list_noise[i] for i in range(len(list_fitness))])

        # compute update step
        # gradient_estimate = np.mean(np.array(list_weighted_noise), axis=0)
        # update_step = gradient_estimate*(self.lr/self.sigma)
        #
        # print("GRADIENT SHAPE", gradient_estimate.shape, "UPDATE_STEP", update_step.shape)
        #
        # # update chromosome
        # self.chromosome[random_indices] = (self.chromosome[random_indices] - update_step)
        # fitness_update = compute_fitness(self.chromosome, self.udp)
        #
        # print(f"The updated solution's fitness is {fitness_update}")

    def fit(self) -> None:

        for epoch in range(self.epochs):
            print(f"COMPUTING FOR EPOCH {epoch}")

            self.update()



