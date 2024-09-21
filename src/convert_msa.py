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
        dataset = NPuzzleDataset(data_path, transform_action=False)
    elif args.env == "sokoban":
        dataset = SokobanDataset(data_path, transform_action=False)
    elif args.env == "minecraft":
        dataset = MinecraftDataset(data_path, transform_action=False)
    else:
        raise ValueError

    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=dataset.collate_fn)

    state_dict = torch.load(args.ckpt)
    n_hidden, n_latent = state_dict["pre_attention.0.weight"].shape
    # msa = MarkovStateAbstraction(input_dims=[("objects", 9216)],
    #                              action_dim=4,
    #                              n_hidden=n_hidden,
    #                              n_latent=n_latent,
    #                              action_classification_type="softmax").to(args.device)
    msa = MarkovStateAbstraction(input_dims=[("agent", 3072),
                                             ("inventory", 3072),
                                             ("objects", 3077)],
                                 action_dim=402,
                                 n_hidden=n_hidden,
                                 n_latent=n_latent,
                                 action_classification_type="sigmoid").to(args.device)
    msa.load_state_dict(state_dict)
    msa.eval()

    n_batch = len(dataset)
    max_obj = max([len(x["objects"])+10 for x in dataset._state])
    # max_obj = max([len(x["objects"]) for x in dataset._state])

    state = np.zeros((n_batch, max_obj, n_latent), dtype=np.float32)
    option = np.zeros((n_batch,), dtype=object)
    next_state = np.zeros_like(state)
    mask = np.zeros_like(state, dtype=bool)

    it = 0
    for x, o, x_ in loader:
        with torch.no_grad():
            z, z_ = msa.encode([x, x_])

        z_all = z.cpu().numpy()
        z_all_ = z_.cpu().numpy()
        diffs = np.linalg.norm(z_all - z_all_, axis=-1)
        m = diffs > 1e-4

        size, n_obj, _ = z_all.shape
        state[it:(it+size), :n_obj] = z_all
        next_state[it:(it+size), :n_obj] = z_all_
        option[it:(it+size)] = [o_i[0] for o_i in o]
        mask[it:(it+size), :n_obj] = m.reshape(size, n_obj, 1)
        it += size

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
