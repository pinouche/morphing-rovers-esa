import pickle
import os

import torch
from torch.utils.data import Dataset

from morphing_rovers.src.autoencoder.utils import create_mode_views_dataset

N_SAMPLES = 1000
VAL_SIZE = 0.2
DATA_PATH_TRAIN = "./training_dataset/train_mode_view_dataset.p"
DATA_PATH_VAL = "./training_dataset/val_mode_view_dataset.p"
device = "cuda" if torch.cuda.is_available() else "cpu"


class CompetitionDataset(Dataset):

    def __init__(self, options, mode='train'):

        data_exists = os.path.exists(DATA_PATH_TRAIN) and os.path.exists(DATA_PATH_VAL)

        if data_exists:
            self.input_data = pickle.load(open(f"./training_dataset/{mode}_mode_view_dataset.p", "rb"))
        else:
            create_mode_views_dataset(options, N_SAMPLES, VAL_SIZE)
            self.input_data = pickle.load(open(f"./training_dataset/{mode}_mode_view_dataset.p", "rb"))

        print("DATASET SHAPE", self.input_data.shape)

        self.filenames = dict()
        self.filenames['x'] = self.input_data
        self.filenames['y'] = self.input_data

    def __getitem__(self, index):
        batch = dict()
        batch['x'] = torch.from_numpy(self.filenames['x'][index])
        batch['y'] = torch.from_numpy(self.filenames['y'][index])

        return batch

    def __len__(self):
        return len(self.filenames['x'])