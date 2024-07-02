import argparse
import os

import torch

from environments.sokoban import MNISTSokoban


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true")
    parser.add_argument("-o", "--output", type=str, default="out")
    parser.add_argument("-n", "--num", type=int, default=1000)
    args = parser.parse_args()

    env = MNISTSokoban(size=(5, 5), max_crates=2, max_steps=50, object_centric=False,
                       render_mode="human", rand_digits=args.random, rand_agent=args.random, rand_x=args.random)

    state = torch.zeros((args.num, 160, 160), dtype=torch.float)
    action = torch.zeros((args.num, 9), dtype=torch.int)
    next_state = torch.zeros((args.num, 160, 160), dtype=torch.float)
    eye = torch.eye(9, dtype=torch.int)

    it = 0
    while it < args.num:
        obs = env.reset()
        done = False
        while not done:
            state[it] = torch.tensor(obs / 255.0, dtype=torch.float)

            a = int(env.sample_action())
            obs, rew, done, info = env.step(a)

            action[it] = eye[a]
            next_state[it] = torch.tensor(obs / 255.0, dtype=torch.float)
            it += 1
            if it >= args.num:
                break

    os.makedirs(args.output, exist_ok=True)
    torch.save(state, f"{args.output}/state.pt")
    torch.save(action, f"{args.output}/action.pt")
    torch.save(next_state, f"{args.output}/next_state.pt")
