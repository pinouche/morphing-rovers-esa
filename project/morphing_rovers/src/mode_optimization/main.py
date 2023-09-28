# import argparse
#
# from utils import get_init_masks_from_path
# from morphing_rovers.src.mode_optimization.optimization.optimization import OptimizeMask
#
#
# if __name__ == "__main__":
#     options = argparse.ArgumentParser(description='Model config')
#     options.add_argument('--config', type=str, default='', help='Path of the config file')
#     options = options.parse_args()
#
#     cluster_trainer_output = get_init_masks_from_path(options)
#
#     mask_optimizer = OptimizeMask(options, data=cluster_trainer_output)
#     mask_optimizer.train()
