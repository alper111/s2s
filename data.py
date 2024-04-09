from typing import NamedTuple

import numpy as np

S2SDataset = NamedTuple('S2SDataset', [
    ('state', np.ndarray),
    ('option', np.ndarray),
    ('reward', np.ndarray),
    ('next_state', np.ndarray),
    ('mask', np.ndarray),
])
