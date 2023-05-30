import torch
import yaml
import numpy as np

from torch.optim import Adam

from morphing_rovers.src.neural_network_supervised.morphing_udp_modified import morphing_rover_UDP, Rover, MAX_DA
from morphing_rovers.utils import Config


class OptimizeNetworkSupervised:

    def __init__(self, options, chromosome):
        self.chromosome = chromosome
        self.options = options

        self.udp = morphing_rover_UDP()
        self.udp.rover = Rover(self.chromosome)
        # self.udp.rover.Control.parameters = chromosome

        self.optimiser = None
        self.loss = float('inf')

        # initialize dataset for nn optimization
        self.rover_view = []
        self.rover_state = []
        self.latent_state = []
        self.data_y = []

        # completed scenarios
        self.completed_scenarios = 0

        ##########
        # Initialise/restore
        ##########
        self.config = None
        config_path = self.options.config
        # Load config file, save it to the experiment output path, and convert to a Config class.
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)

    def reset_data(self):
        self.udp.rover.training_data = []
        self.udp.rover.cluster_data = []

    def load_data(self, n_iter):

        self.udp.fitness(self.udp.rover, self.completed_scenarios, n_iter)

        # print("LEN OF TRAINING DATA", len(self.udp.rover.training_data))

        for index in range(len(self.udp.rover.training_data)):

            # collect the training data by running the simulation for the current chromosome
            controller_input = self.udp.rover.training_data[index][0]
            target = self.udp.rover.training_data[index][1][1]

            if target < -np.pi/4:
                target = -np.pi/4
            elif target > np.pi/4:
                target = np.pi/4
            else:
                target = target

            self.rover_view.append(np.squeeze(controller_input[0]))
            self.rover_state.append(np.squeeze(controller_input[1]))
            self.latent_state.append(np.squeeze(controller_input[2]))
            self.data_y.append(target)

    def create_optimizer(self):

        self.optimiser = Adam(list(self.udp.rover.Control.parameters()),
                              self.config.learning_rate_supervised_learning)

    def loss_function(self, angular_change, target):
        angular_adjustment = MAX_DA * angular_change
        loss = (angular_adjustment - torch.tensor(target)) ** 2
        return loss

    def activate_gradient(self):
        self.udp.rover.Control.requires_grad_(True)

    def train_step(self):
        # self.activate_gradient()

        _, angular_change, _ = self.udp.rover.Control(torch.unsqueeze(torch.from_numpy(np.stack(self.rover_view)), dim=1),
                                                      torch.from_numpy(np.stack(self.rover_state)),
                                                      torch.from_numpy(np.stack(self.latent_state)))

        loss = self.loss_function(angular_change, self.data_y).mean()

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def train(self, n_iter, train=True):
        self.reset_data()
        self.create_optimizer()
        self.load_data(n_iter)

        if train:
            for iteration_step in range(self.config.n_iter_supervised_learning):
                loss = self.train_step()

                if (iteration_step+1) % 10 == 0:
                   print(f"Computing for iteration number {iteration_step+1}")
                   print(f"The average loss is: {loss}")
