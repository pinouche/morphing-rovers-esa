import yaml
import pickle
import numpy as np
import torch
import os

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from morphing_rovers.utils import Config
from morphing_rovers.src.clustering.utils import load_checkpoint

DATA_PATH_TRAIN = "./autoencoder/training_dataset/train_mode_view_dataset.p"
DATA_PATH_VAL = "./autoencoder/training_dataset/val_mode_view_dataset.p"
PCA_MODEL = "./clustering/experiments/pca.p"

K = 10


class ClusteringTerrain:

    def __init__(self, options, data=None, groupby_scenario = False, random_state=None):
        self.options = options
        self.model = None

        self.views = None
        self.scenarios_id = None
        self.groupby_scenario = groupby_scenario
        self.random_state = random_state
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
            self.views = torch.unsqueeze(torch.stack([d[0] for d in self.data]), 1)

            if self.groupby_scenario:
                self.scenarios_id = np.array([d[-1] for d in self.data])
                self.data = torch.stack([self.views[self.scenarios_id == i].mean(dim=0) for i in range(30)])

            else:
                self.data = self.views

        # get latent representation
        self.latent_representation = self.model.encoder(self.data).numpy(force=True)

    def run(self):

        self.load_trained_autoencoder()
        self.get_latent_representation()

        if os.path.exists(PCA_MODEL):
            pca_model = pickle.load(open(PCA_MODEL, "rb"))
        else:
            if self.groupby_scenario:
                raise ValueError("pca model does not exists")
            else:
                pca_model = PCA(n_components=50)
                pca_model.fit(self.latent_representation)
                pickle.dump(pca_model, open(PCA_MODEL, "wb"))

        self.latent_representation = pca_model.transform(self.latent_representation)[:, :K]

        if self.config.clustering_algo == "kmeans":
            cluster_model = KMeans(n_clusters=self.config.n_clusters, random_state=self.random_state, n_init="auto")
            clusters = cluster_model.fit_predict(self.latent_representation)

        elif self.config.clustering_algo == "gmm":
            cluster_model = GaussianMixture(n_components=self.config.n_clusters, random_state=self.random_state)
            clusters = cluster_model.fit_predict(self.latent_representation)

        else:
            raise ValueError(f"clustering algo {self.config.clustering_algo} not supported.")

        print("CLUSTERS COUNTS", np.unique(clusters, return_counts=True))

        if self.groupby_scenario:
            scenarios = np.arange(0, 30, 1)
            dict_replace = dict(zip(scenarios, clusters))
            clusters = np.array([dict_replace[k] for k in self.scenarios_id])

        self.output = [self.views, clusters]
        # pickle.dump((self.data, self.latent_representation, clusters), open("./experiments/clusters.p", "wb"))
