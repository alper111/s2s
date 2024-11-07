import pickle
import os

import torch
import numpy as np


class HanoiDataset(torch.utils.data.Dataset):
    ACTION_TO_IDX = {
        "left-middle": 1,
        "left-right": 2,
        "middle-left": 3,
        "middle-right": 4,
        "right-left": 5,
        "right-middle": 6,
    }

    def __init__(self, root_folder: str, transform_action: bool = True,
                 **kwargs):
        self._root_folder = root_folder
        self._transform_action = transform_action
        self._state = np.load(os.path.join(root_folder, "state.npy"), allow_pickle=True)
        self._next_state = np.load(os.path.join(root_folder, "next_state.npy"), allow_pickle=True)
        self._action = pickle.load(open(os.path.join(root_folder, "action.pkl"), "rb"))

    def __getitem__(self, idx):
        x = torch.tensor(self._state[idx], dtype=torch.float)
        x_ = torch.tensor(self._next_state[idx], dtype=torch.float)
        if self._transform_action:
            a = self._actions_to_label(self._action[idx])
        else:
            a = self._action[idx]
        return x, a, x_

    def __len__(self):
        return len(self._state)

    def sample(self, n_samples):
        idx = np.random.choice(len(self), n_samples, replace=True)
        batch = [self[i] for i in idx]
        return self.collate_fn(batch)

    @staticmethod
    def collate_fn(batch):
        x = torch.stack([b[0] for b in batch])
        if isinstance(batch[0][1], torch.Tensor):
            a = torch.cat([b[1] for b in batch])
        else:
            a = [b[1] for b in batch]
        x_ = torch.stack([b[2] for b in batch])
        return x, a, x_

    @staticmethod
    def _actions_to_label(action):
        a = torch.tensor([HanoiDataset.ACTION_TO_IDX[action]], dtype=torch.long)
        return a
