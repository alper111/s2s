import os
import pickle

import torch
import numpy as np

from abstraction.msa import MarkovStateAbstraction
from environments.minecraft import MinecraftDataset
from environments.sokoban import SokobanDataset
from s2s.structs import S2SDataset


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
        train_config = self.config["abstraction"]["training"]
        loader = self._get_loader(batch_size=train_config["batch_size"],
                                  exclude_keys=["global"])
        save_path = os.path.join(self.config["save_path"], "abstraction")
        self.abstraction.fit(loader, train_config, save_path)

    def load_abstraction(self):
        path = os.path.join(self.config["save_path"], "abstraction")
        self.abstraction.load(path)

    def convert_with_abstraction(self, mask_threshold=1e-4, batch_size=100):
        loader = self._get_loader(batch_size=batch_size, transform_action=False, shuffle=False)
        self.load_abstraction()
        n_sample = len(loader.dataset)
        keys = self.abstraction.order
        max_obj = max([sum([len(x[k]) for k in keys]) for x in loader.dataset._state])

        n_latent = self.config["abstraction"]["parameters"]["n_latent"]
        state = np.zeros((n_sample, max_obj, n_latent), dtype=np.float32)
        state_global = []
        option = np.zeros((n_sample,), dtype=object)
        next_state = np.zeros_like(state)
        next_state_global = []
        mask = np.zeros_like(state, dtype=bool)
        mask_global = []

        it = 0
        for s, o, sn in loader:
            with torch.no_grad():
                (z, zn) = self.abstraction.encode([s, sn])

            z = z.cpu().numpy()
            zn = zn.cpu().numpy()
            diffs = np.linalg.norm(z - zn, axis=-1)
            m = diffs > mask_threshold

            s_global = s["global"].flatten(1, -1).numpy()
            sn_global = sn["global"].flatten(1, -1).numpy()
            m_global = np.abs(s_global - sn_global) > mask_threshold

            size, n_obj, _ = z.shape
            state[it:(it+size), :n_obj] = z
            next_state[it:(it+size), :n_obj] = zn
            option[it:(it+size)] = o
            mask[it:(it+size), :n_obj] = m.reshape(size, n_obj, 1)

            state_global.append(s_global)
            next_state_global.append(sn_global)
            mask_global.append(m_global)

            it += size

        state_global = np.concatenate(state_global)
        next_state_global = np.concatenate(next_state_global)
        mask_global = np.concatenate(mask_global)

        dataset = S2SDataset(state, option, np.zeros(option.shape), next_state, mask)
        dataset_global = S2SDataset(state_global, option, np.zeros(option.shape), next_state_global, mask_global)
        local_data_path = os.path.join(self.config["save_path"], "datasets")
        os.makedirs(local_data_path, exist_ok=True)
        save_file = open(os.path.join(local_data_path, "abs_dataset.pkl"), "wb")
        save_file_global = open(os.path.join(local_data_path, "global.pkl"), "wb")
        pickle.dump(dataset, save_file)
        pickle.dump(dataset_global, save_file_global)
        return dataset, dataset_global

    def _get_loader(self, batch_size, transform_action=True, exclude_keys=[], shuffle=True):
        datapath = os.path.join("data", self.env)
        if self.env == "sokoban":
            dataset_class = SokobanDataset
        elif self.env == "minecraft":
            dataset_class = MinecraftDataset
        else:
            raise ValueError

        dataset = dataset_class(datapath,
                                transform_action=transform_action,
                                exclude_keys=exclude_keys)

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             collate_fn=dataset.collate_fn)
        return loader
