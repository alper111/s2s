import torch


class MarkovStateAbstraction(torch.nn.Module):
    def __init__(self, input_dims: list[tuple], action_dim: int, n_hidden: int, n_latent: int,
                 action_classification_type: str = "softmax"):
        super(MarkovStateAbstraction, self).__init__()
        self._order = [x[0] for x in input_dims]
        self._cls_type = action_classification_type
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
        )

        self.pre_attention = torch.nn.Sequential(
            torch.nn.Linear(2*n_latent, n_hidden),
            torch.nn.ReLU()
        )
        _att_layers = torch.nn.TransformerEncoderLayer(d_model=n_hidden, nhead=4, batch_first=True)
        self.attention = torch.nn.TransformerEncoder(_att_layers, num_layers=4)
        self.ctx_embedding = torch.nn.Embedding(2, n_hidden)

        self.inverse_fc = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, action_dim)
        )
        self.density_fc = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def inverse_forward(self, h):
        a_logits = self.inverse_fc(h)
        return a_logits

    def density_forward(self, h):
        y_logits = self.density_fc(h)
        return y_logits[:, 0, 0]

    def encode(self, x):
        return {k: self.feature(self.projection[k](x[k].to(self.device))) for k in x if k != "masks"}

    def attn_forward(self, z, z_, pad_mask):
        # flatten dictionaries to tensors
        zf = self._flatten(z)
        zf_ = self._flatten(z_)
        # combine state and next state tensors
        n_pad = zf.shape[1] - zf_.shape[1]
        if n_pad > 0:
            zf_ = torch.cat([zf_, torch.zeros(zf_.shape[0], n_pad, zf_.shape[2], device=self.device)], dim=1)
        elif n_pad < 0:
            zf_ = zf_[:, :zf.shape[1], :]
        z_cat = torch.cat([zf, zf_], dim=-1)
        z_proj = self.pre_attention(z_cat)
        n_batch, n_token, _ = z_proj.shape
        obj_idx = torch.zeros(n_batch, n_token, dtype=torch.long, device=self.device)
        cls_idx = torch.ones(n_batch, 1, dtype=torch.long, device=self.device)
        # add classification token to the beginning of the sequence
        z_proj = z_proj + self.ctx_embedding(obj_idx)
        z_proj = torch.cat([self.ctx_embedding(cls_idx), z_proj], dim=1)
        # pad mask to account for classification token
        mask = self._flatten(pad_mask).to(self.device)
        mask = torch.cat([torch.ones(n_batch, 1, dtype=torch.bool, device=self.device), mask], dim=1)
        # apply attention
        h = self.attention(z_proj, src_key_padding_mask=~mask)
        # just to ensure there is no accidental backprop
        h = h * mask.unsqueeze(2)
        return h

    def forward(self, x):
        z = self.encode(x)
        return z

    def inverse_loss(self, h, a, mask):
        m = self._flatten(mask).to(self.device)
        a_logits = self.inverse_forward(h)
        if self._cls_type == "softmax":
            a_logits = a_logits.permute(0, 2, 1)
            loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device), reduction="none")
        elif self._cls_type == "sigmoid":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a_logits, a.to(self.device), reduction="none")
            loss = loss.sum(dim=-1)
        else:
            raise ValueError(f"Unknown action classification type: {self._cls_type}")
        loss = (loss * m).sum() / m.sum()
        return loss

    def density_loss(self, h_pos, h_neg):
        n_batch = h_pos.shape[0]
        h = torch.cat([h_pos, h_neg], dim=0)
        y = torch.cat([torch.ones(n_batch), torch.zeros(n_batch)], dim=0).to(self.device)
        y_logits = self.density_forward(h)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_logits, y)

    def regularization(self, z, z_):
        z = self._flatten(z)
        z_ = self._flatten(z_)
        l1_loss = torch.nn.functional.l1_loss(z, z_)
        return l1_loss

    def loss(self, x, x_, x_neg, a):
        z = self.forward(x)
        z_ = self.forward(x_)
        z_neg = self.forward(x_neg)

        h_pos = self.attn_forward(z, z_, x["masks"])
        # we probably need harder negative examples
        h_neg = self.attn_forward(z, z_neg, x["masks"])

        inv_loss = self.inverse_loss(h_pos[:, 1:], a, x["masks"])
        density_loss = self.density_loss(h_pos, h_neg)
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
