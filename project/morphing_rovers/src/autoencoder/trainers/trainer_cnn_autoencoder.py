import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from morphing_rovers.src.autoencoder.trainers.trainer import Trainer
from morphing_rovers.src.autoencoder.dataset.torch_dataset import CompetitionDataset
from morphing_rovers.src.autoencoder.models.model import Autoencoder
from morphing_rovers.src.autoencoder.metrics.reconstruction import ReconstructionMetric


class TerrainTrainer(Trainer):
    def create_data(self):
        self.train_dataset = CompetitionDataset(self.options, mode='train')
        self.val_dataset = CompetitionDataset(self.options, mode='val')

        self.train_dataloader = DataLoader(self.train_dataset, self.config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, self.config.batch_size, shuffle=False)

    def create_model(self):
        self.model = Autoencoder(self.config.encoded_space_dim, self.config.fc2_input_dim)

    def create_loss(self):
        self.loss_fn = nn.MSELoss()

    def create_optimiser(self):
        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad, self.config.learning_rate, weight_decay=self.config.weight_decay)

    def create_metrics(self):
        self.train_metrics = ReconstructionMetric('train')
        self.val_metrics = ReconstructionMetric('val')

    def forward_model(self, batch):
        return self.model(batch['x'])

    def forward_loss(self, batch, output):
        return self.loss_fn(output, batch['y'])
