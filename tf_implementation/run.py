import yaml

from typing import Dict

from tensorflow.python.framework import ops
from single_hidden_layer_net import SingleHiddenLayerNet


def parse_yaml(path_to_file: str) -> Dict:
    with open(path_to_file) as file:
        data = dict()
        try:
            data = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError:
            pass
        return data


def main():
    ops.reset_default_graph()
    cfg = parse_yaml('tf_implementation/net_config.yaml')
    net = SingleHiddenLayerNet(**cfg)
    net.train()


if __name__ == '__main__':
    main()
