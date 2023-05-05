import yaml
import pickle
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from morphing_rovers.utils import Config
from morphing_rovers.src.clustering.utils import load_checkpoint

DATA_PATH_TRAIN = "./autoencoder/training_dataset/train_mode_view_dataset.p"
DATA_PATH_VAL = "./autoencoder/training_dataset/val_mode_view_dataset.p"


class ClusteringTerrain:

    def __init__(self, options, data=None):
        self.options = options
        self.model = None

        self.data = data
        self.latent_representation = None
        self.output = None

        ##########
        # Initialise/restore
        ##########
        self.config = None
        config_path = self.options.config
        # Load config file, save it to the experiment output path, and convert to a Config class.
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config = Config(self.config)

    def load_trained_autoencoder(self):
        self.model = load_checkpoint(self.config.session_name, self.config.encoded_space_dim, self.config.fc2_input_dim)

    def get_latent_representation(self):
        # load data

        if self.data is None:
            train_data = pickle.load(open(DATA_PATH_TRAIN, "rb"))
            val_data = pickle.load(open(DATA_PATH_VAL, "rb"))
            self.data = torch.Tensor(np.expand_dims(np.concatenate((train_data, val_data)), 1))

        else:
            self.data = torch.unsqueeze(torch.stack([d[0] for d in self.data]), 1)

        # get latent representation
        self.latent_representation = self.model.encoder(self.data)

    def run(self):

        self.load_trained_autoencoder()
        self.get_latent_representation()

        pca_model = PCA(n_components=20)
        self.latent_representation = pca_model.fit_transform(self.latent_representation.numpy(force=True))

        if self.config.clustering_algo == "kmeans":
            cluster_model = KMeans(n_clusters=self.config.n_clusters, random_state=1, n_init="auto")
            clusters = cluster_model.fit_predict(self.latent_representation)

        elif self.config.clustering_algo == "gmm":
            cluster_model = GaussianMixture(n_components=self.config.n_clusters, random_state=1)
            clusters = cluster_model.fit_predict(self.latent_representation)

        else:
            raise ValueError(f"clustering algo {self.config.clustering_algo} not supported.")

        self.output = [self.data, clusters]
        # pickle.dump((self.data, self.latent_representation, clusters), open("./experiments/clusters.p", "wb"))
