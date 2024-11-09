import torch

from s2s.structs import FlatDataset


class HanoiDataset(FlatDataset):
    ACTION_TO_IDX = {
        "left-middle": 1,
        "left-right": 2,
        "middle-left": 3,
        "middle-right": 4,
        "right-left": 5,
        "right-middle": 6,
    }

    @staticmethod
    def _actions_to_label(action):
        a = torch.tensor([HanoiDataset.ACTION_TO_IDX[action]], dtype=torch.long)
        return a
