import os
import argparse

import torch

from abstraction.msa import MarkovStateAbstraction
from environments.minecraft import MinecraftDataset
from environments.npuzzle import NPuzzleDataset
from environments.sokoban import SokobanDataset


def main(args):
    data_path = os.path.join("data", args.env)
    if args.env == "npuzzle":
        dataset = NPuzzleDataset(data_path)
    elif args.env == "sokoban":
        dataset = SokobanDataset(data_path)
    elif args.env == "minecraft":
        dataset = MinecraftDataset(data_path)
    else:
        raise ValueError

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=dataset.collate_fn, num_workers=8)

    msa = MarkovStateAbstraction(input_dims=[("agent", 3072),
                                             ("inventory", 3072),
                                             ("objects", 3077)],
                                 action_dim=402,
                                 n_hidden=args.n_hidden,
                                 n_latent=args.n_latent,
                                 action_classification_type="sigmoid").to(args.device)
    optimizer = torch.optim.Adam(msa.parameters(), lr=args.lr)

    for e in range(args.epoch):
        avg_inv_loss = 0
        avg_density_loss = 0
        avg_reg_loss = 0
        for x, a, x_ in loader:
            n = x["objects"].shape[0]
            x_n, _, _ = dataset.sample(n)
            inv_loss, density_loss, reg_loss = msa.loss(x, x_, x_n, a)
            loss = inv_loss + density_loss  # + 0.01*reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_inv_loss += inv_loss.item()
            avg_density_loss += density_loss.item()
            avg_reg_loss += reg_loss.item()
        print(f"Epoch {e + 1}/{args.epoch}, inv_loss={avg_inv_loss / len(loader):.5f}, "
              f"density_loss={avg_density_loss / len(loader):.5f}, "
              f"smoothness_loss={avg_reg_loss / len(loader):.5f}")

        if (e+1) % 20 == 0:
            save_folder = os.path.join("save", args.env)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{args.out}.pt")
            torch.save(msa.cpu().state_dict(), save_path)
            msa.to(args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Markov state abstractions")
    parser.add_argument("--env", type=str, help="The environment name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument("--n-hidden", type=int, default=256, help="Number of hidden units")
    parser.add_argument("--n-latent", type=int, default=16, help="Number of latent units")
    parser.add_argument("--out", type=str, help="Output name")
    args = parser.parse_args()
    main(args)
