import yaml
import pickle
import numpy as np
import torch
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering

from morphing_rovers.utils import Config
from morphing_rovers.src.clustering.utils import load_checkpoint, swap_values, compute_velocity_matrix

DATA_PATH_TRAIN = "./autoencoder/training_dataset/train_mode_view_dataset.p"
DATA_PATH_VAL = "./autoencoder/training_dataset/val_mode_view_dataset.p"
PCA_MODEL = "./clustering/experiments/pca.p"

USE_VELOCITY = True
K = 3


class ClusteringTerrain:

    def __init__(self, options, path_data=None, groupby_scenario=False, algorithm=None, random_state=None):
        self.options = options
        self.model = None

        self.views = None
        self.scenarios_id = None
        self.groupby_scenario = groupby_scenario
        self.random_state = random_state
        self.data = path_data
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

        if self.config.clustering_algo == "None":
            self.config.clustering_algo = algorithm

    def load_trained_autoencoder(self):
        self.model = load_checkpoint(self.config.session_name, self.config.encoded_space_dim, self.config.fc2_input_dim)

    def get_latent_representation(self):
        # load data

        if self.data is None:
            train_data = pickle.load(open(DATA_PATH_TRAIN, "rb"))
            val_data = pickle.load(open(DATA_PATH_VAL, "rb"))
            self.data = torch.Tensor(np.expand_dims(np.concatenate((train_data, val_data)), 1))

        else:
            self.views = torch.stack([d[0] for d in self.data])
            self.views = torch.unsqueeze(self.views, dim=1)

            if self.groupby_scenario:
                self.scenarios_id = np.array([d[-1] for d in self.data])
                self.data = torch.stack([self.views[self.scenarios_id == i].mean(dim=0) for i in np.unique(self.scenarios_id)])

            else:
                self.data = self.views

        # # get latent representation
        # pickle.dump(self.data, open("data_views.p", "wb"))

        if not USE_VELOCITY:
            self.latent_representation = self.model.encoder(self.data).numpy(force=True)

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
            self.latent_representation *= pca_model.explained_variance_ratio_[:K]
        else:
            self.latent_representation = compute_velocity_matrix(self.data)

    def run(self):

        self.load_trained_autoencoder()
        self.get_latent_representation()

        if self.config.clustering_algo == "kmeans":
            cluster_model = KMeans(n_clusters=self.config.n_clusters, random_state=self.random_state, init="k-means++",
                                   n_init="auto")
            clusters = cluster_model.fit_predict(self.latent_representation)

        elif self.config.clustering_algo == "gmm":
            cluster_model = GaussianMixture(n_components=self.config.n_clusters, init_params='k-means++',
                                            random_state=self.random_state)
            cluster_model.fit(self.latent_representation)
            clusters = cluster_model.predict(self.latent_representation)

        elif self.config.clustering_algo == "agg":
            metric = None
            if USE_VELOCITY:
                metric = "precomputed"
            cluster_model = AgglomerativeClustering(n_clusters=self.config.n_clusters, linkage='complete', metric=metric)
            # compute_full_tree=True, distance_threshold=0.1

            clusters = cluster_model.fit_predict(self.latent_representation)

        elif self.config.clustering_algo == "manual":
            clusters = np.ones(30)*0
            clusters[13] = 1
            clusters[14] = 2
            clusters[16] = 3

            # [0, 4, 6, 7, 8, 12, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29]
            # [1, 2, 3, 10, 11, 18, 20]
            # [9, 15, 19]
            # [16]

        else:
            raise ValueError(f"clustering algo {self.config.clustering_algo} not supported.")

        clusters = swap_values(clusters)
        print("CLUSTERS COUNTS", np.unique(clusters, return_counts=True))

        self.output = [self.views, self.data, clusters]
