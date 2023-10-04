from argparse import Namespace
import yaml


def load_config(config_file):
    """Load config."""
    with open(config_file) as yaml_file:
        configs = yaml.safe_load(yaml_file)
    return dict(configs)


class Config(Namespace):
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)
