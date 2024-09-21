import os

import torch
import torch.utils

from abstraction.msa import MarkovStateAbstraction
from environments.minecraft import MinecraftDataset
from environments.sokoban import SokobanDataset


class Agent:
    def __init__(self, config):
        self.config = config
        self.env = config["env"]
        os.makedirs(config["save_path"], exist_ok=True)

        if config["abstraction"]["method"] == "msa":
            self.abstraction = MarkovStateAbstraction(config["abstraction"]["parameters"])
            self.abstraction.to(self.config["abstraction"]["training"]["device"])
        elif config["abstraction"]["method"] == "ae":
            raise NotImplementedError
        elif config["abstraction"]["method"] == "pca":
            raise NotImplementedError

    def train_abstraction(self):
        loader = self._get_loader(exclude_keys=["global"])
        save_path = os.path.join(self.config["save_path"], "abstraction")
        self.abstraction.fit(loader, self.config["abstraction"]["training"], save_path)

    def load_abstraction(self):
        path = os.path.join(self.config["save_path"], "abstraction")
        self.abstraction.load(path)

    def _get_loader(self, exclude_keys=[]):
        datapath = os.path.join("data", self.env)
        if self.env == "sokoban":
            dataset = SokobanDataset(datapath, exclude_keys=exclude_keys)
        elif self.env == "minecraft":
            dataset = MinecraftDataset(datapath, exclude_keys=exclude_keys)
        else:
            raise ValueError

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.config["abstraction"]["training"]["batch_size"],
                                             shuffle=True,
                                             collate_fn=dataset.collate_fn)
        return loader
