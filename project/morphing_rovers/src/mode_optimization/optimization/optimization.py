import yaml
import pickle
import numpy as np
import torch
from torch.optim import Adam

from morphing_rovers.src.mode_optimization.utils import load_data, velocity_function
from morphing_rovers.src.utils import Config

device = "cuda" if torch.cuda.is_available() else "cpu"


class OptimizeMask:

    def __init__(self, options, solution_list=None, data=None):
        self.options = options

        self.optimiser = None
        self.velocity = float('inf')*-1

        # data variables
        self.data = data
        self.mode_view_data = []
        self.cluster_id = []

        # store the solution
        self.solution = None
        self.solution_list = solution_list
        self.optimized_masks = []
        self.weighted_average = None

        ##########
        # Initialise/restore
        ##########
        self.config = None
        config_path = self.options.config
        # Load config file, save it to the experiment output path, and convert to a Config class.
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)

    def load_training_data(self):
        if self.data is None:
            self.data = load_data()

    def prepare_data(self):
        self.mode_view_data = torch.squeeze(self.data[0])
        self.cluster_id = self.data[1]

    def initialize_solution(self):
        if self.solution_list is None:
            self.solution_list = []
            for _ in range(4):
                self.solution_list.append(torch.rand(11, 11, requires_grad=True))

    def create_optimizer(self):
        self.optimiser = Adam([self.solution], self.config.learning_rate_mask_optimization)

    def train_step(self, solution_expand, batch):
        velocity = velocity_function(solution_expand, batch).mean()
        velocity *= -1

        self.optimiser.zero_grad()
        velocity.backward()
        self.optimiser.step()

        return velocity.item()

    def train(self):
        weighted_average_velocity = 0
        self.prepare_data()
        self.initialize_solution()
        for cluster_id in range(4):

            self.solution = self.solution_list[cluster_id]
            self.create_optimizer()

            data_cluster = self.mode_view_data[self.cluster_id == cluster_id]

            if data_cluster.shape[0] > 0:
                # print(f"Training for cluster {cluster_id} with {data_cluster.shape[0]} sample.")
                for iteration_step in range(self.config.n_iter_mask_optimization):
                    solution_expand = self.solution.repeat(data_cluster.shape[0], 1, 1)
                    self.velocity = self.train_step(solution_expand, data_cluster)

                    # print("the current average velocity over the all cluster is: {:.3f}".format(self.velocity))

                weighted_average_velocity += self.velocity*data_cluster.shape[0]

            self.optimized_masks.append(self.solution)

        self.weighted_average = weighted_average_velocity/len(self.data[0])
        # pickle.dump(self.optimized_masks, open("./experiments/optimized_masks.p", "wb"))