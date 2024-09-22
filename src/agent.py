import os
import pickle

import torch
import numpy as np

from abstraction.msa import MarkovStateAbstraction
from environments.minecraft import MinecraftDataset
from environments.sokoban import SokobanDataset
from s2s.structs import S2SDataset
from s2s.factorise import factors_from_partitions
from s2s.partition import partition_to_subgoal
from s2s.vocabulary import (build_vocabulary, merge_partitions_by_map, build_schemata,
                            build_typed_schemata, append_to_schemata)


class Agent:
    def __init__(self, config):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s]: %(message)s',
                            datefmt="%H:%M:%S", force=True)
        self.logger = logging.getLogger("main")

        self.env = config["env"]
        self.save_path = config["save_path"]
        self.abstraction_method = config["abstraction"]["method"]
        self.abstraction_params = config["abstraction"]["parameters"]
        self.train_config = config["abstraction"]["training"]
        self.s2s_config = config["s2s"]
        self.s2s_g_config = config["s2s_global"]
        self.env = config["env"]
        os.makedirs(config["save_path"], exist_ok=True)

        if self.abstraction_method == "msa":
            self.abstraction = MarkovStateAbstraction(self.abstraction_params)
            self.abstraction.to(self.train_config["device"])
        elif self.abstraction_method == "ae":
            raise NotImplementedError
        elif self.abstraction_method == "pca":
            raise NotImplementedError

    def fit_s2s(self):
        path = os.path.join(self.save_path, "s2s")
        path_g = os.path.join(self.save_path, "s2s_g")
        os.makedirs(path, exist_ok=True)
        os.makedirs(path_g, exist_ok=True)

        dataset, dataset_g = self._get_abstract_dataset()
        partition_config = self.s2s_config["partition"]
        partitions, partitions_g = partition_to_subgoal(dataset, dataset_g,
                                                        eps=partition_config["eps"],
                                                        mask_threshold=partition_config["mask_threshold"])
        self.logger.info(f"Number of partitions={len(partitions)}")

        self.logger.info("Finding factors")
        factors = factors_from_partitions(partitions, threshold=self.s2s_config["factor_threshold"])
        factors_g = factors_from_partitions(partitions_g, threshold=self.s2s_g_config["factor_threshold"])
        self.logger.info(f"Number of factors={len(factors)}, {len(factors_g)}")

        self.logger.info("Building vocabulary")
        res = build_vocabulary(partitions, factors, "s",
                               density_type=self.s2s_config["density_type"],
                               comparison=self.s2s_config["comparison"],
                               factor_threshold=self.s2s_config["factor_threshold"],
                               independency_test=self.s2s_config["independency_test"],
                               k_cross=self.s2s_config["k_cross"],
                               pre_threshold=self.s2s_config["pre_threshold"],
                               min_samples_split=self.s2s_config["min_samples_split"],
                               pos_threshold=self.s2s_config["pos_threshold"])
        vocabulary, pre_props, eff_props, merge_map = res
        self.vocabulary = vocabulary
        _dump(vocabulary, os.path.join(path, "vocabulary.pkl"))
        _dump(pre_props, os.path.join(path, "pre_props.pkl"))
        _dump(eff_props, os.path.join(path, "eff_props.pkl"))
        self.logger.info(f"Vocabulary size={len(vocabulary)}")

        self.logger.info("Merging global partitions w.r.t. lifted partitions")
        partitions_g = merge_partitions_by_map(partitions_g, merge_map)
        global_extended = {}
        for p_i in partitions_g:
            parts_i, _ = partition_to_subgoal(partitions_g[p_i])
            for p_ij in parts_i:
                global_extended[p_i + p_ij[1:]] = parts_i[p_ij]
        partitions_g = global_extended

        self.logger.info("Building global vocabulary")
        res = build_vocabulary(partitions_g, factors_g, "p",
                               density_type=self.s2s_g_config["density_type"],
                               comparison=self.s2s_g_config["comparison"],
                               factor_threshold=self.s2s_g_config["factor_threshold"],
                               independency_test=self.s2s_g_config["independency_test"],
                               k_cross=self.s2s_g_config["k_cross"],
                               pre_threshold=self.s2s_g_config["pre_threshold"],
                               min_samples_split=self.s2s_g_config["min_samples_split"],
                               pos_threshold=self.s2s_g_config["pos_threshold"])
        vocabulary_g, pre_props_g, eff_props_g, _ = res
        self.vocabulary_g = vocabulary_g
        _dump(vocabulary_g, os.path.join(path_g, "vocabulary.pkl"))
        _dump(pre_props_g, os.path.join(path_g, "pre_props.pkl"))
        _dump(eff_props_g, os.path.join(path_g, "eff_props.pkl"))
        self.logger.info(f"Global vocabulary size={len(vocabulary_g)}")

        self.logger.info("Building schemata")
        schemata = build_schemata(vocabulary, pre_props, eff_props)
        schemata_g = build_schemata(vocabulary_g, pre_props_g, eff_props_g)
        schemata_t, types, groups, mapping, object_types = build_typed_schemata(vocabulary, schemata)
        _dump(schemata, os.path.join(path, "schemata.pkl"))
        _dump(schemata_g, os.path.join(path_g, "schemata.pkl"))
        _dump(schemata_t, os.path.join(path, "typed_schemata.pkl"))
        _dump(types, os.path.join(path, "types.pkl"))
        _dump(groups, os.path.join(path, "groups.pkl"))
        _dump(mapping, os.path.join(path, "mapping.pkl"))
        _dump(object_types, os.path.join(path, "object_types.pkl"))

        schemata_a = append_to_schemata(schemata_t, schemata_g)
        _dump(schemata_a, os.path.join(path, "schemata_a.pkl"))
        self.logger.info(f"Number of action schemas={len(schemata_a)}.")

    def train_abstraction(self):
        loader = self._get_loader(batch_size=self.train_config["batch_size"],
                                  exclude_keys=["global"])
        save_path = os.path.join(self.save_path, "abstraction")
        self.abstraction.fit(loader, self.train_config, save_path)

    def load_abstraction(self):
        self.logger.info("Loading abstraction model")
        path = os.path.join(self.save_path, "abstraction")
        self.abstraction.load(path)

    def convert_with_abstraction(self, mask_threshold=1e-4, batch_size=100):
        loader = self._get_loader(batch_size=batch_size, transform_action=False, shuffle=False)
        self.load_abstraction()
        n_sample = len(loader.dataset)
        keys = self.abstraction.order
        max_obj = max([sum([len(x[k]) for k in keys]) for x in loader.dataset._state])

        n_latent = self.abstraction_params["n_latent"]
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
        local_data_path = os.path.join(self.save_path, "datasets")
        os.makedirs(local_data_path, exist_ok=True)

        _dump(dataset, os.path.join(local_data_path, "abs_dataset.pkl"))
        _dump(dataset_global, os.path.join(local_data_path, "global.pkl"))
        return dataset, dataset_global

    def _get_abstract_dataset(self, mask_threshold=1e-4, batch_size=100):
        local_data_path = os.path.join(self.save_path, "datasets")
        data_file = os.path.join(local_data_path, "abs_dataset.pkl")
        data_file_global = os.path.join(local_data_path, "global.pkl")
        if os.path.exists(data_file) and os.path.exists(data_file_global):
            self.logger.info("Found existing datasets, loading")
            dataset = pickle.load(open(data_file, "rb"))
            dataset_global = pickle.load(open(data_file_global, "rb"))
        else:
            self.logger.info("Converting dataset with abstraction")
            dataset, dataset_global = self.convert_with_abstraction(mask_threshold, batch_size)
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


def _dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
