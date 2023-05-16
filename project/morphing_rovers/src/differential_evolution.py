import numpy as np
import pickle
from scipy.optimize import differential_evolution

from morphing_rovers.morphing_udp import morphing_rover_UDP
from morphing_rovers.src.utils import fitness_wrapper

# parameters
POP_SIZE = 100
SIGMA = 0.0001
MAXITER = 10

N_PARAMETERS = 19126
N_HYPERPARAMETERS = 7


if __name__ == "__main__":

    chromosome = pickle.load(open("./trained_chromosomes/chromosome_fitness_2.0049.p", "rb"))

    udp = morphing_rover_UDP()
    func = udp.fitness
    print(f"the fitness of the original champion is {func(chromosome)}")

    # create random noise
    pop_array = np.transpose(np.repeat(np.expand_dims(chromosome, 1), POP_SIZE, 1))

    noise = np.random.randn(POP_SIZE, 19126)*SIGMA
    pop_array[:, :N_PARAMETERS] = pop_array[:, :N_PARAMETERS] + noise
    lower_bounds = np.min(pop_array, axis=0) - np.array([SIGMA] * N_PARAMETERS + [0] * N_HYPERPARAMETERS)
    higher_bounds = np.max(pop_array, axis=0) + np.array([SIGMA] * (N_PARAMETERS + N_HYPERPARAMETERS))
    bounds = list(zip(lower_bounds, higher_bounds))

    result = differential_evolution(func=fitness_wrapper, bounds=bounds, maxiter=MAXITER, init=pop_array,
                                    polish=False, recombination=0, tol=0.001, vectorized=False, seed=1)

    print(result)
