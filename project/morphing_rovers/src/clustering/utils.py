import os
import torch

from morphing_rovers.src.autoencoder.models.model import Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(session_name: str = "session_1", latent_dim: int = 50, fc_dim: int = 256) -> Autoencoder:

    checkpoint_name = os.path.join("..", "autoencoder", "experiments", session_name)
    checkpoint = torch.load(checkpoint_name, map_location=device)

    model = Autoencoder(latent_dim, fc_dim)
    model.load_state_dict(checkpoint['model'])

    print('Loaded model and optimiser weights from {}\n'.format(checkpoint_name))

    return model

