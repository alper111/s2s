import os
import pickle
import argparse

import torch
import numpy as np

from environments.minecraft import MinecraftDataset
from environments.npuzzle import NPuzzleDataset
from environments.sokoban import SokobanDataset
from s2s.structs import S2SDataset


def main(args):
    data_path = os.path.join("data", args.env)
    if args.env == "npuzzle":
        dataset = NPuzzleDataset(data_path, transform_action=False)
    elif args.env == "sokoban":
        dataset = SokobanDataset(data_path, transform_action=True, privileged=True)
    elif args.env == "minecraft":
        dataset = MinecraftDataset(data_path, transform_action=False)
    elif args.env == "mc_priv":
        dataset = MinecraftDataset(data_path, transform_action=False, privileged=True)
    else:
        raise ValueError

    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=dataset.collate_fn)

    state = []
    state_global = []
    option = []
    next_state = []
    next_state_global = []
    mask = []
    mask_global = []
    n_max = 0
    for s, o, sn in loader:
        # START MINECRAFT SPECIFIC
        # xa = torch.cat([torch.full((x["agent"].shape[0], x["agent"].shape[1], 1), fill_value=1), x["agent"]], axis=-1)
        # xi = torch.cat([torch.full((x["inventory"].shape[0], x["inventory"].shape[1], 1), fill_value=2), x["inventory"]], axis=-1)
        # xo = torch.cat([torch.full((x["objects"].shape[0], x["objects"].shape[1], 1), fill_value=3), x["objects"]], axis=-1)
        # xa_ = torch.cat([torch.full((x_["agent"].shape[0], x_["agent"].shape[1], 1), fill_value=1), x_["agent"]], axis=-1)
        # xi_ = torch.cat([torch.full((x_["inventory"].shape[0], x_["inventory"].shape[1], 1), fill_value=2), x_["inventory"]], axis=-1)
        # xo_ = torch.cat([torch.full((x_["objects"].shape[0], x_["objects"].shape[1], 1), fill_value=3), x_["objects"]], axis=-1)
        # x = torch.cat([xa, xi, xo], axis=1)
        # x_ = torch.cat([xa_, xi_, xo_], axis=1)
        # END MINECRAFT SPECIFIC

        s_local = s["objects"].numpy()
        sn_local = sn["objects"].numpy()
        s_global = s["global"].flatten(1, -1).numpy()
        sn_global = sn["global"].flatten(1, -1).numpy()

        n_max = max(n_max, s_local.shape[1])
        m_local = np.abs(s_local - sn_local) > 1e-8
        m_global = np.abs(s_global - sn_global) > 1e-8

        state.append(s_local)
        option.append([o_i[0] for o_i in o])
        next_state.append(sn_local)
        mask.append(m_local)
        state_global.append(s_global)
        next_state_global.append(sn_global)
        mask_global.append(m_global)

    for i in range(len(state)):
        s = state[i]
        o = option[i]
        sn = next_state[i]
        m = mask[i]
        n_rem = n_max - s.shape[1]
        if n_rem > 0:
            state[i] = np.concatenate([s, np.zeros((s.shape[0], n_rem, s.shape[2]))], axis=1)
            # option[i] = np.concatenate([o, np.zeros((o.shape[0], n_rem, o.shape[2]))], axis=1)
            next_state[i] = np.concatenate([sn, np.zeros((sn.shape[0], n_rem, sn.shape[2]))], axis=1)
            mask[i] = np.concatenate([m, np.zeros((m.shape[0], n_rem, m.shape[2]))], axis=1)

    state = np.concatenate(state)
    option = np.concatenate(option)
    next_state = np.concatenate(next_state)
    mask = np.concatenate(mask)
    state_global = np.concatenate(state_global)
    next_state_global = np.concatenate(next_state_global)
    mask_global = np.concatenate(mask_global)

    dataset = S2SDataset(state, option, np.zeros(option.shape[0]), next_state, mask)
    dataset_global = S2SDataset(state_global, option, np.zeros(option.shape[0]), next_state_global, mask_global)
    save_file = open(os.path.join("data", args.env, "abs_dataset.pkl"), "wb")
    save_file_global = open(os.path.join("data", args.env, "global.pkl"), "wb")
    pickle.dump(dataset, save_file)
    pickle.dump(dataset_global, save_file_global)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert dataset with MSA model")
    parser.add_argument("--env", type=str, help="The environment name")
    args = parser.parse_args()
    main(args)
