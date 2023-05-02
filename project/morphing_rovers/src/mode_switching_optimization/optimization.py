import torch
import yaml
import numpy as np
import os
import pickle

from torch.optim import Adam

from morphing_udp_modified import morphing_rover_UDP, Rover
from morphing_rovers.src.utils import Config


class OptimizeNetworkSupervisedSwitching:

    def __init__(self, options, chromosome):
        self.chromosome = chromosome
        self.options = options

        self.udp = morphing_rover_UDP()
        self.udp.rover = Rover(self.chromosome)

        self.optimiser = None
        self.loss = float('inf')

        # initialize dataset
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

    def load_data(self, n_iter):

        if os.path.exists("mode_switching_dataset.p"):
            training_data = pickle.load(open("mode_switching_dataset.p", "rb"))
        else:
            self.udp.fitness(self.udp.rover, n_iter)
            training_data = self.udp.rover.training_data
            pickle.dump(training_data, open("mode_switching_dataset.p", "wb"))

        print("LEN OF TRAINING DATA", len(training_data))

        for index in range(len(training_data)):

            data = training_data[index]
            # collect the training data by running the simulation for the current chromosome
            target = data[-1]

            if target:
                target = 1
            else:
                target = -1

            self.rover_view.append(np.squeeze(data[0]))
            self.rover_state.append(np.squeeze(data[1]))
            self.latent_state.append(np.squeeze(data[2]))
            self.data_y.append(target)

        print(f"Random classification gives an accuracy of {np.sum(np.array(self.data_y) == 1)/len(self.data_y)}")

    def create_optimizer(self):

        self.optimiser = Adam(list(self.udp.rover.Control.parameters()),
                              self.config.learning_rate_supervised_learning_mode_switching)

    def accuracy_metric(self, prediction, target):
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = -1
        accuracy = np.sum(prediction.numpy(force=True) == np.array(target))/len(target)

        return accuracy

    def loss_function(self, prediction, target):
        loss = torch.abs((prediction - torch.tensor(target)))
        # loss = binary_cross_entropy(prediction, torch.tensor(target))
        return loss

    def train_step(self):
        switching_mode, _, _ = self.udp.rover.Control(torch.unsqueeze(torch.from_numpy(np.stack(self.rover_view)), dim=1),
                                                      torch.from_numpy(np.stack(self.rover_state)),
                                                      torch.from_numpy(np.stack(self.latent_state)))

        loss = self.loss_function(switching_mode, self.data_y).mean()
        accuracy = self.accuracy_metric(switching_mode, self.data_y)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item(), accuracy

    def train(self, n_iter):
        self.reset_data()
        self.create_optimizer()
        self.load_data(n_iter)

        for iteration_step in range(self.config.n_iter_supervised_learning_mode_switching):
            loss, accuracy = self.train_step()

            if iteration_step % 10 == 0:
                print(f"Computing for iteration number {iteration_step+1}")
                print(f"The average loss is: {loss}")
                print(f"The accuracy is: {accuracy}")
