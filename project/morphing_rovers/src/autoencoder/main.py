import argparse
from morphing_rovers.src.autoencoder.trainers.trainer_cnn_autoencoder import TerrainTrainer


if __name__ == '__main__':
    options = argparse.ArgumentParser(description='Model config')
    options.add_argument('--config', type=str, default='', help='Path of the config file')
    options = options.parse_args()

    trainer = TerrainTrainer(options)
    trainer.train()
