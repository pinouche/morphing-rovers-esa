import numpy as np
import copy
import yaml
import random

from morphing_rovers.src.evolution_strategies.utils import get_noise, perturb_chromosome, compute_fitness
from morphing_rovers.morphing_udp import morphing_rover_UDP
from morphing_rovers.utils import Config


N_PARAM_TO_PERTURB = 19126


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
        print(f"The current fitness is {self.best_fitness}")

        list_noise = []
        list_fitness = []
        # compute fitness for each network in the population
        for p in range(self.pop_size):
            random_indices = random.sample(range(N_PARAM_TO_PERTURB), 100)
            temporary_chromosome = copy.deepcopy(self.chromosome)

            if p % 10 == 0:
                print(f"Computing fitness for individual number {p}")

            # get the noise
            noise = get_noise(len(random_indices))

            chromosome_to_perturb = copy.deepcopy(temporary_chromosome[random_indices])
            chromosome_to_perturb = perturb_chromosome(chromosome_to_perturb, noise, self.sigma)
            temporary_chromosome[random_indices] = chromosome_to_perturb

            # compute the fitness
            f_obj = compute_fitness(temporary_chromosome, self.udp)[0]
            list_fitness.append(f_obj)
            list_noise.append(noise)

            if f_obj < self.best_fitness:
                print(f"new best fitness is {f_obj}")
                self.chromosome = temporary_chromosome
                self.best_fitness = f_obj

        # list_weighted_noise = np.array([list_fitness[i]*list_noise[i] for i in range(len(list_fitness))])

        # self.chromosome = self.best_chromosome

        # compute update step
        # gradient_estimate = np.mean(np.array(list_weighted_noise), axis=0)
        # update_step = gradient_estimate*(self.lr/self.sigma)
        #
        # print("GRADIENT SHAPE", gradient_estimate.shape, "UPDATE_STEP", update_step.shape)
        #
        # # update chromosome
        # self.chromosome[:N_PARAM_TO_PERTURB] = (self.chromosome[:N_PARAM_TO_PERTURB] - update_step)
        # fitness_update = compute_fitness(self.chromosome, self.udp)

        # print(f"The updated solution's fitness is {fitness_update}")

    def fit(self) -> None:

        for epoch in range(self.epochs):
            print(f"COMPUTING FOR EPOCH {epoch}")

            self.update()



