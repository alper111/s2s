class Abstraction:
    def __init__(self, config):
        self.config = config

    def fit(self):
        raise NotImplementedError

    def encode(self, x, **kwargs):
        raise NotImplementedError

    def decode(self, z, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
