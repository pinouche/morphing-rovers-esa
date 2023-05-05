import numpy as np
import copy

from utils import get_noise, perturb_chromosome, compute_fitness

from morphing_rovers.morphing_udp import morphing_rover_UDP


N_PARAM_TO_PERTURB = 19126


class Solution:

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.udp = morphing_rover_UDP()

    def update(self, sigma: float, lr: float, pop_size: int) -> None:

        temporary_chromosome = copy.deepcopy(self.chromosome)
        current_fitness = compute_fitness(temporary_chromosome, self.udp)
        print(f"The current fitness is {current_fitness}")

        fitness_list = list()
        fitness_list.append(current_fitness)

        list_weighted_noise = []
        # compute fitness for each network in the population
        for p in range(pop_size):
            if p % 10 == 0:
                print(f"Computing fitness for individual number {p}")

            # get the noise
            noise = get_noise(N_PARAM_TO_PERTURB)
            perturbed_chromosome = perturb_chromosome(temporary_chromosome, noise, sigma)

            # compute the fitness
            f_obj = compute_fitness(perturbed_chromosome, self.udp)
            fitness_list.append(f_obj)
            weighted_noise = f_obj*noise
            list_weighted_noise.append(weighted_noise)

        # compute update step
        gradient_estimate = np.mean(np.array(list_weighted_noise), axis=0)
        update_step = [grad*(lr/sigma) for grad in gradient_estimate]

        # update chromosome
        self.chromosome += update_step
        fitness_update = compute_fitness(self.chromosome, self.udp)

        print(f"The champion's fitness is {np.max(fitness_list)}, and the updated solution's fitness is {fitness_update}")

    def fit(self, sigma: float, learning_rate: float, pop_size: int, epochs: int) -> None:

        objective_list = []

        best_objective_val = 0
        for epoch in range(epochs):
            print(f"COMPUTING FOR EPOCH {epoch}")

            self.update(sigma, learning_rate, pop_size)

