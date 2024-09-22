import os
import pickle
import logging
import subprocess

import torch
import numpy as np

from abstraction.msa import MarkovStateAbstraction
from environments.minecraft import MinecraftDataset
from environments.sokoban import SokobanDataset
from s2s.structs import S2SDataset, PDDLDomain, PDDLProblem
from s2s.factorise import factors_from_partitions
from s2s.partition import partition_to_subgoal
from s2s.vocabulary import (build_vocabulary, merge_partitions_by_map, build_schemata,
                            build_typed_schemata, append_to_schemata)
from s2s.helpers import dict_to_tensordict


class Agent:
    def __init__(self, config):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s]: %(message)s',
                            datefmt="%H:%M:%S", force=True)
        self.logger = logging.getLogger("main")

        self.env = config["env"]
        self.name = config["name"]
        self.save_path = os.path.join("save", self.env, self.name)
        self.abstraction_method = config["abstraction"]["method"]
        self.abstraction_params = config["abstraction"]["parameters"]
        self.train_config = config["abstraction"]["training"]
        self.s2s_config = config["s2s"]
        self.s2s_g_config = config["s2s_global"]
        self._fd_path = config["fast_downward_path"]

        os.makedirs(self.save_path, exist_ok=True)

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
        _dump(vocabulary_g, os.path.join(path_g, "vocabulary.pkl"))
        _dump(pre_props_g, os.path.join(path_g, "pre_props.pkl"))
        _dump(eff_props_g, os.path.join(path_g, "eff_props.pkl"))
        self.logger.info(f"Global vocabulary size={len(vocabulary_g)}")

        self.logger.info("Building schemata")
        schemata = build_schemata(vocabulary, pre_props, eff_props)
        schemata_g = build_schemata(vocabulary_g, pre_props_g, eff_props_g)
        schemata_t, types, groups, mapping, object_types = build_typed_schemata(vocabulary, schemata)
        schemata_a = append_to_schemata(schemata_t, schemata_g)
        _dump(schemata, os.path.join(path, "schemata.pkl"))
        _dump(schemata_g, os.path.join(path_g, "schemata.pkl"))
        _dump(schemata_t, os.path.join(path, "typed_schemata.pkl"))
        _dump(schemata_a, os.path.join(path, "schemata_a.pkl"))
        _dump(types, os.path.join(path, "types.pkl"))
        _dump(groups, os.path.join(path, "groups.pkl"))
        _dump(mapping, os.path.join(path, "mapping.pkl"))
        _dump(object_types, os.path.join(path, "object_types.pkl"))
        self.logger.info(f"Number of action schemas={len(schemata_a)}.")
        self.sym_to_type = mapping
        self.object_types = object_types
        self.domain = PDDLDomain(f"{self.env}_{self.name}", [vocabulary, vocabulary_g], schemata_a, types)

    def load_s2s(self):
        path = os.path.join(self.save_path, "s2s")
        path_g = os.path.join(self.save_path, "s2s_g")
        vocabulary = _load(os.path.join(path, "vocabulary.pkl"))
        vocabulary_g = _load(os.path.join(path_g, "vocabulary.pkl"))
        schemata_a = _load(os.path.join(path, "schemata_a.pkl"))
        types = _load(os.path.join(path, "types.pkl"))
        mapping = _load(os.path.join(path, "mapping.pkl"))
        object_types = _load(os.path.join(path, "object_types.pkl"))
        self.sym_to_type = mapping
        self.object_types = object_types
        self.domain = PDDLDomain(f"{self.env}_{self.name}", [vocabulary, vocabulary_g], schemata_a, types)

    def get_abstract_vector(self, state, key_order=None):
        x, key_order = dict_to_tensordict(state, exclude_keys=["global"], key_order=key_order)
        with torch.no_grad():
            z = self.abstraction.encode(x)
        return z, key_order

    def get_symbols(self, state, state_g=None, key_order=None, from_raw=True):
        if from_raw:
            if "global" in state:
                state_g = state["global"][0].astype(np.float32)
            state, key_order = self.get_abstract_vector(state, key_order)
            state = state.cpu().numpy()
        symbols = self.domain.active_symbols(state, state_g)
        return symbols, key_order

    def get_symbol_grounding(self, symbol, modality, n=100):
        z = torch.tensor(symbol.sample(n), dtype=torch.float32, device=self.abstraction.device)
        z = z.unsqueeze(1)
        with torch.no_grad():
            idx = self.abstraction.order.index(modality)
            tokens = [0 for _ in range(idx)]
            tokens.append(1)
            x = self.abstraction.decode(z, tokens)
            x = x[modality].squeeze().cpu().numpy()
        return x

    def initialize_problem(self, state, goal, problem_name="p1"):
        p_init, order = self.get_symbols(state)
        p_goal, _ = self.get_symbols(goal, key_order=order)
        obj_types = {}
        for obj in p_init:
            if obj == "global":
                continue
            f_types = {self.sym_to_type[p] for p in p_init[obj]}
            for otype in self.object_types:
                if self.object_types[otype] == f_types:
                    obj_types[obj] = otype
                    break
        problem = PDDLProblem(problem_name, self.domain.name)
        problem.initialize_from_dict(p_init, p_goal, obj_types)
        return problem

    def plan(self, state, goal, problem_name="p1", method="ff", verbose=False, raw_actions=False):
        if method == "ff":
            args = ["--evaluator", "hff=ff()", "--search", "lazy_greedy([hff], preferred=[hff])"]
        elif method == "astar":
            args = ["--search", "astar(blind())"]

        problem = self.initialize_problem(state, goal, problem_name)
        domain_file = os.path.join(self.save_path, "domain.pddl")
        problem_file = os.path.join(self.save_path, f"{problem_name}.pddl")
        print(self.domain, file=open(domain_file, "w"))
        print(problem, file=open(problem_file, "w"))
        if os.path.exists("sas_plan"):
            os.remove("sas_plan")
        args = [self._fd_path, domain_file, problem_file] + args
        out = None if verbose else subprocess.DEVNULL
        subprocess.run(args, stdout=out, stderr=out)
        plan = None
        if os.path.exists("sas_plan"):
            with open("sas_plan", "r") as f:
                plan = f.read()
                plan = plan.split("\n")[:-2]
                if not raw_actions:
                    plan = [x.strip("()") for x in plan]
                    plan = [x.split(" ") for x in plan]
                    plan = [(x[0].split("_")[1], x[1:]) for x in plan]
        return plan

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
            dataset = _load(data_file)
            dataset_global = _load(data_file_global)
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


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
