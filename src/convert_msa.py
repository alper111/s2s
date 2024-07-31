import os
import pickle
import argparse

import torch
import numpy as np

from abstraction.msa import MarkovStateAbstraction
from environments.minecraft import MinecraftDataset
from environments.npuzzle import NPuzzleDataset
from environments.sokoban import SokobanDataset
from s2s.structs import S2SDataset


def main(args):
    data_path = os.path.join("data", args.env)
    if args.env == "npuzzle":
        dataset = NPuzzleDataset(data_path)
    elif args.env == "sokoban":
        raise SokobanDataset(data_path)
    elif args.env == "minecraft":
        dataset = MinecraftDataset(data_path)
    else:
        raise ValueError

    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=dataset.collate_fn)

    state_dict = torch.load(args.ckpt)
    n_hidden, n_latent = state_dict["pre_attention.0.weight"].shape
    msa = MarkovStateAbstraction(input_dims=[("objects", 786)],
                                 action_dim=4,
                                 n_hidden=n_hidden,
                                 n_latent=n_latent).to(args.device)
    msa.eval()

    state = []
    option = []
    next_state = []
    mask = []
    for x, o, x_ in loader:
        with torch.no_grad():
            z = msa.encode(x)
            z_ = msa.encode(x_)
            z_all = z["objects"].cpu().numpy().round().astype(int)
            z_all_ = z_["objects"].cpu().numpy().round().astype(int)
            m = np.abs(z_all - z_all_) > 1e-8
            state.append(z_all)
            option.append(o)
            next_state.append(z_all_)
            mask.append(m)
    state = np.concatenate(state)
    option = np.concatenate(option)
    next_state = np.concatenate(next_state)
    mask = np.concatenate(mask)

    dataset = S2SDataset(state, option, np.zeros(option.shape), next_state, mask)
    save_file = open(os.path.join("data", args.env, "abs_dataset.pkl"), "wb")
    pickle.dump(dataset, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert dataset with MSA model")
    parser.add_argument("--env", type=str, help="The environment name")
    parser.add_argument("--ckpt", type=str, help="Model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    args = parser.parse_args()
    main(args)
