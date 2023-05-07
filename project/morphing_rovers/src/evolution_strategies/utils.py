import numpy as np


def get_noise(size):
    noise = np.random.randn(size)
    return noise


def perturb_chromosome(chromosome, noise, sigma):
    chromosome_tmp = chromosome + noise * sigma
    return chromosome_tmp


def compute_fitness(chromosome, udp):
    fitness = udp.fitness(chromosome)
    return fitness



