import numpy as np
import pickle
import matplotlib.pyplot as plt

CLUSTER_DATA_PATH = "clusters.p"

data = pickle.load(open(CLUSTER_DATA_PATH, "rb"))

print(np.unique(data[-1], return_counts=True))

cluster = data[0][data[-1] == 3]
latent_dim = data[1][data[-1] == 3]

for i in range(20):
    fig, axes = plt.subplots(ncols=4, figsize=(8, 4))

    ax1, ax2, ax3, ax4 = axes

    im1_org = ax1.matshow(np.squeeze(cluster[i]))
    im2_org = ax2.matshow(np.squeeze(cluster[i+1]))

    im1_latent = ax3.matshow(np.repeat(np.expand_dims(latent_dim[i], 0), 50, 0))
    im2_latent = ax4.matshow(np.repeat(np.expand_dims(latent_dim[i+1], 0), 50, 0))

    fig.colorbar(im1_org, ax=ax1)
    fig.colorbar(im2_org, ax=ax2)
    fig.colorbar(im1_latent, ax=ax3)
    fig.colorbar(im2_latent, ax=ax4)

    plt.show()
    plt.close()
