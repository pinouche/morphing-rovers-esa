from abc import ABCMeta, abstractmethod

import json
import os
import pickle
import yaml
import torch
from tqdm import tqdm

from morphing_rovers.utils import Config


device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    __meta_class__ = ABCMeta

    def __init__(self, options):
        self.options = options
        ##########
        # Trainer utils
        ##########
        self.global_step = 0
        self.start_time = None
        self.best_score = float('inf')  # if the higher, the better times by -1
        # to log results
        self.loss_per_epoch = []
        self.reconstructed_data = []

        ##########
        # Initialise/restore
        ##########
        self.config = None
        config_path = self.options.config
        # Load config file, save it to the experiment output path, and convert to a Config class.
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)

        ##########
        # Data
        ##########
        self.train_dataset, self.val_dataset = None, None
        self.train_dataloader, self.val_dataloader = None, None
        self._train_dataloader_iter = None
        self.create_data()

        ##########
        # Model
        ##########
        self.device = device
        self.model = None
        self.create_model()
        self.model.to(self.device)

        ##########
        # Loss
        ##########
        self.loss_fn = None
        self.create_loss()

        ##########
        # Optimiser
        ##########
        self.optimiser = None
        self.create_optimiser()

        ##########
        # Metrics
        ##########
        self.train_metrics = None
        self.val_metrics = None
        self.create_metrics()

    @abstractmethod
    def create_data(self):
        """Create train/val datasets and dataloaders."""

    @abstractmethod
    def create_model(self):
        """Build the neural network."""

    @abstractmethod
    def create_loss(self):
        """Build the loss function."""

    @abstractmethod
    def create_optimiser(self):
        """Create the model's optimiser."""

    @abstractmethod
    def create_metrics(self):
        """Implement the metrics."""

    @abstractmethod
    def forward_model(self, batch):
        """Compute the output of the model."""

    @abstractmethod
    def forward_loss(self, batch, output):
        """Compute the loss."""

    def train_step(self, batch, iteration):
        # Fetch data
        # batch = self._get_next_batch()
        self.preprocess_batch(batch)

        # Forward pass
        output = self.forward_model(batch)
        loss = self.forward_loss(batch, output)

        # Backward pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def train(self):
        print('Starting training session..')
        self.model.train()

        for epoch in range(self.config.n_epochs):
            if (epoch+1) % 50 == 0:
                print(f"Epoch number {epoch}")
            train_loss = 0
            for iteration, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):

                loss = self.train_step(batch, iteration)
                train_loss += loss

            train_loss /= len(self.train_dataloader)

            print("training loss is: {:.3f}".format(train_loss))

            test_loss = self.test()
            self.loss_per_epoch.append((train_loss, test_loss))
            self.global_step += 1

            if test_loss < self.best_score:
                print("new best loss of: {:.3f}".format(test_loss))
                self.best_score = test_loss
                self.save_checkpoint()

        with open("./experiments/results_loss.json", "w") as f:
            dic_result = {"loss": self.loss_per_epoch}
            json.dump(dic_result, f)

        # with open("./experiments/results_reconstruction.p", "wb") as f:
        #     pickle.dump(self.reconstructed_data, f)

    def test_step(self, batch, iteration):
        self.preprocess_batch(batch)
        output = self.forward_model(batch)
        # self.reconstructed_data.append((batch['x'].numpy(force=True), output.numpy(force=True)))
        loss = self.forward_loss(batch, output)

        return loss.item()

    def test(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for iteration, batch in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                loss = self.test_step(batch, iteration)
                val_loss += loss

            val_loss /= len(self.val_dataloader)

        print(f'Val loss: {val_loss:.4f}')

        self.model.train()
        return val_loss

    def _get_next_batch(self):
        if self._train_dataloader_iter is None:
            self._train_dataloader_iter = iter(self.train_dataloader)
        batch = None
        while batch is None:
            try:
                batch = next(self._train_dataloader_iter)
            except StopIteration:
                self._train_dataloader_iter = iter(self.train_dataloader)
        return batch

    def preprocess_batch(self, batch):
        # Cast to device
        for key, value in batch.items():
            batch[key] = torch.unsqueeze(value.to(self.device), 1)

    def save_checkpoint(self):
        checkpoint = dict(model=self.model.state_dict(),
                          optimiser=self.optimiser.state_dict(),
                          )

        checkpoint_name = os.path.join(".", "experiments", self.config.session_name)
        torch.save(checkpoint, checkpoint_name)
        print('Model saved to: {}\n'.format(checkpoint_name))

    def load_checkpoint(self):
        checkpoint_name = os.path.join(".", "experiments", self.config.session_name)
        checkpoint = torch.load(checkpoint_name, map_location=device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimiser.load_state_dict(checkpoint['optimiser'])
        print('Loaded model and optimiser weights from {}\n'.format(checkpoint_name))
