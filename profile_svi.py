import argparse
import logging
import os

import torch

from pyrophylo.models import CountyModel

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(relativeCreated) 9d %(message)s',
                    level=logging.DEBUG)


def load_dataset(args):
    assert os.path.exists("results/model_inputs.pt"), \
        "missing results/model_inputs.pt try running model_1.ipynb"
    dataset = torch.load("results/model_inputs.pt")
    logger.info("loaded {} trees".format(len(dataset["trees"])))
    assert args.num_trees <= len(dataset["trees"])
    dataset["trees"] = dataset["trees"][:args.num_trees]
    return dataset


def main(args):
    dataset = load_dataset(args)
    model = CountyModel(**dataset)
    model.fit_svi(num_particles=args.num_particles,
                  num_steps=args.num_steps,
                  log_every=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--num-trees", type=int, default=100)
    parser.add_argument("-n", "--num-steps", type=int, default=10)
    parser.add_argument("-p", "--num-particles", type=int, default=1)
    args = parser.parse_args()
    main(args)
