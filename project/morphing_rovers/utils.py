from argparse import Namespace


class Config(Namespace):
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)
