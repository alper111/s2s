import torch


class MarkovStateAbstraction(torch.nn.Module):
    def __init__(self, input_dims: tuple, action_dim: int, n_hidden: int, n_latent: int):
        super(MarkovStateAbstraction, self).__init__()
        self._order = [x[0] for x in input_dims]
        self.projection = torch.nn.ModuleDict(
            {key: torch.nn.Sequential(
                torch.nn.Linear(value, n_hidden),
                torch.nn.ReLU())
             for (key, value) in input_dims})
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_latent),
            GumbelSigmoidLayer()
        )
        self.pre_attention = torch.nn.Sequential(
            torch.nn.Linear(n_latent, n_hidden),
            torch.nn.ReLU()
        )
        _att_layers = torch.nn.TransformerEncoderLayer(d_model=n_hidden, nhead=4, batch_first=True)
        self.attention = torch.nn.TransformerEncoder(_att_layers, num_layers=4)
        self.inverse = torch.nn.Sequential(
            torch.nn.Linear(2*n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, action_dim)
        )
        self.density_fc = torch.nn.Sequential(
            torch.nn.Linear(2*n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def inverse_forward(self, z, z_):
        z_all = torch.cat([z, z_], dim=-1)
        a_logits = self.inverse(z_all)
        return a_logits

    def density_forward(self, z, z_, z_n):
        z_pos = torch.cat([z, z_], dim=-1)
        z_neg = torch.cat([z, z_n], dim=-1)
        z_all = torch.cat([z_pos, z_neg], dim=0)
        y_logits = self.density_fc(z_all).squeeze()
        return y_logits

    def encode(self, x):
        return {k: self.feature(self.projection[k](x[k].to(self.device))) for k in x if k != "masks"}

    def aggregate(self, z, pad_mask):
        zf = self._flatten(z)
        mask = self._flatten(pad_mask).to(self.device)
        z_proj = self.pre_attention(zf)
        z_agg = self.attention(z_proj, src_key_padding_mask=~mask)
        z_agg = (z_agg * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1).unsqueeze(1)
        return z_agg

    def forward(self, x):
        z = self.encode(x)
        z_agg = self.aggregate(z, x["masks"])
        return z, z_agg

    def inverse_loss(self, z, z_, a):
        a_logits = self.inverse_forward(z, z_)
        loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device))
        return loss

    def density_loss(self, z, z_, z_n):
        n_batch = z.shape[0]
        y_logits = self.density_forward(z, z_, z_n)
        y = torch.cat([torch.ones(n_batch), torch.zeros(n_batch)], dim=0).to(self.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_logits, y)

    def regularization(self, z, z_):
        z = self._flatten(z)
        z_ = self._flatten(z_)
        l1_loss = torch.nn.functional.l1_loss(z, z_)
        return l1_loss

    def loss(self, x, x_, x_neg, a):
        z, z_agg = self.forward(x)
        z_, z_agg_ = self.forward(x_)
        _, zn_agg = self.forward(x_neg)
        inv_loss = self.inverse_loss(z_agg, z_agg_, a)
        density_loss = self.density_loss(z_agg, z_agg_, zn_agg)
        regularization = self.regularization(z, z_)
        return inv_loss, density_loss, regularization

    def _flatten(self, x):
        return torch.cat([x[k] for k in self._order], dim=1)


class GumbelSigmoidLayer(torch.nn.Module):
    def __init__(self, hard=False, T=1.0):
        super(GumbelSigmoidLayer, self).__init__()
        self.hard = hard
        self.T = T

    def forward(self, x):
        if not self.training:
            return torch.sigmoid(x)
        else:
            return gumbel_sigmoid(x, self.T, self.hard)


def sample_gumbel_diff(*shape):
    eps = 1e-20
    u1 = torch.rand(shape)
    u2 = torch.rand(shape)
    diff = torch.log(torch.log(u2+eps)/torch.log(u1+eps)+eps)
    return diff


def gumbel_sigmoid(logits, T=1.0, hard=False):
    g = sample_gumbel_diff(*logits.shape)
    g = g.to(logits.device)
    y = (g + logits) / T
    s = torch.sigmoid(y)
    if hard:
        s_hard = s.round()
        s = (s_hard - s).detach() + s
    return s
