from getmusic.utils.misc import instantiate_from_config


def build_model(config, args=None):
    return instantiate_from_config(config['model'])
