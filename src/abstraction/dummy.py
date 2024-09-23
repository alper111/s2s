import torch

from .base import Abstraction


class Dummy(Abstraction):
    def __init__(self, config):
        Abstraction.__init__(self, config)
        self.config = config
        self.device = "cpu"
        self._order = [x[0] for x in self.config["input_dims"]]

    @property
    def order(self):
        return self._order

    def fit(self, *args, **kwargs):
        pass

    def encode(self, x, **kwargs):
        if isinstance(x, list):
            out = []
            for x_i in x:
                out.append(self._encode_one(x_i))
        else:
            out = self._encode_one(x)
        return out

    def decode(self, z, **kwargs):
        return z

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def _encode_one(self, x):
        out = []
        for mod_i in self.order:
            out.append(x[mod_i])
        out = torch.cat(out, dim=1)
        return out
