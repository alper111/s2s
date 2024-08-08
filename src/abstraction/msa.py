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
            torch.nn.Linear(n_latent, n_hidden),
            torch.nn.ReLU()
        )
        _att_layers = torch.nn.TransformerEncoderLayer(d_model=n_hidden, nhead=4, batch_first=True)
        self.attention = torch.nn.TransformerEncoder(_att_layers, num_layers=4)
        self.context = torch.nn.Embedding(4, n_hidden)

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
    def order(self):
        return self._order

    @property
    def device(self):
        return next(self.parameters()).device

    def inverse_forward(self, h):
        return self.inverse_fc(h)

    def density_forward(self, h):
        return self.density_fc(h)

    def encode(self, x):
        # all this mambo jambo is just to process them in a single forward
        projs = []
        mod_tokens = []
        for mod_i in self.order:
            inputs = []
            tokens = []
            if isinstance(x, list):
                n = len(x)
                for x_i in x:
                    inputs.append(x_i[mod_i])
                    tokens.append(x_i[mod_i].shape[1])
            else:
                n = 1
                inputs.append(x[mod_i])
                tokens.append(x[mod_i].shape[1])
            inputs = torch.cat(inputs, dim=1).to(self.device)
            proj_i = self.projection[mod_i](inputs)
            projs.append(proj_i)
            mod_tokens.append(tokens)
        projs = torch.cat(projs, dim=1)
        feats = self.feature(projs)
        outputs = []
        it = 0
        for tokens in mod_tokens:
            mod_outs = []
            for t_i in tokens:
                mod_outs.append(feats[:, it:(it+t_i)])
                it += t_i
            outputs.append(mod_outs)

        return_feats = []
        for i in range(n):
            f = []
            for out in outputs:
                f.append(out[i])
            f = torch.cat(f, dim=1)
            return_feats.append(f)
        if n == 1:
            return_feats = return_feats[0]
        return return_feats

    def attn_forward(self, z, zn, m, mn):
        n_batch, z_token, _ = z.shape
        _, zn_token, _ = zn.shape
        ctx_density = self.context(torch.full((n_batch, 1), 0, dtype=torch.long, device=self.device))
        ctx_action = self.context(torch.full((n_batch, 1), 1, dtype=torch.long, device=self.device))
        ctx_z = self.context(torch.full((n_batch, z_token), 2, dtype=torch.long, device=self.device))
        ctx_zn = self.context(torch.full((n_batch, zn_token), 3, dtype=torch.long, device=self.device))

        z_all = torch.cat([z, zn], dim=1)
        z_all = self.pre_attention(z_all)
        z, zn = z_all[:, :z_token], z_all[:, z_token:]
        z = z + ctx_z
        zn = zn + ctx_zn
        inputs = torch.cat([ctx_action, ctx_density, z, zn], dim=1)
        mask = torch.cat([torch.ones(n_batch, 2, dtype=torch.bool, device=self.device), m, mn], dim=1)

        h = self.attention(inputs, src_key_padding_mask=~mask)
        # just to ensure there is no accidental backprop
        h = h * mask.unsqueeze(2)
        return h

    def forward(self, x):
        return self.encode(x)

    def inverse_loss(self, h, a, mask):
        a_logits = self.inverse_forward(h)
        if self._cls_type == "softmax":
            assert a.ndim == 2
            a_logits = a_logits.permute(0, 2, 1)
            loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device), reduction="none")
        elif self._cls_type == "sigmoid":
            assert a.ndim == 3
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a_logits, a.to(self.device), reduction="none")
            loss = loss.sum(dim=-1)
        else:
            raise ValueError(f"Unknown action classification type: {self._cls_type}")
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def density_loss(self, h, y):
        y_logits = self.density_forward(h).flatten()
        return torch.nn.functional.binary_cross_entropy_with_logits(y_logits, y)

    def regularization(self, z, z_):
        l1_loss = torch.nn.functional.l1_loss(z, z_)
        return l1_loss

    def loss(self, x, x_, x_neg, a):
        z, z_, z_neg = self.forward([x, x_, x_neg])
        n_batch, n_pos, n_dim = z.shape
        n_neg = z_neg.shape[1]

        xm = self._flatten(x["masks"]).to(self.device)
        xm_ = self._flatten(x_["masks"]).to(self.device)
        xm_neg = self._flatten(x_neg["masks"]).to(self.device)

        z_init = z.repeat(2, 1, 1)
        m_init = xm.repeat(2, 1)

        # if the positive and negative samples don't have the same
        # number of objects in them...
        n_remaining = n_pos - n_neg
        if n_remaining > 0:
            zeros = torch.zeros(n_batch, n_remaining, n_dim, dtype=torch.float, device=self.device)
            z_neg = torch.cat([z_neg, zeros], dim=1)
            mask_zeros = torch.zeros(n_batch, n_remaining, dtype=torch.bool, device=self.device)
            xm_neg = torch.cat([xm_neg, mask_zeros], dim=1)
        elif n_remaining < 0:
            zeros = torch.zeros((n_batch, -n_remaining, n_dim), dtype=torch.float, device=self.device)
            z_ = torch.cat([z_, zeros], dim=1)
            mask_zeros = torch.zeros(n_batch, -n_remaining, dtype=torch.bool, device=self.device)
            xm_ = torch.cat([xm_, mask_zeros], dim=1)
        z_next = torch.cat([z_, z_neg], dim=0)
        m_next = torch.cat([xm_, xm_neg], dim=0)

        h_all = self.attn_forward(z_init, z_next, m_init, m_next)
        h_density = h_all[:, 0]  # for g(y | z, z') estimation
        y_density = torch.cat([torch.ones(n_batch), torch.zeros(n_batch)], dim=0).to(self.device)
        h_action = h_all[:n_batch, 1:(n_pos+2)]
        a_mask = torch.cat([torch.ones(n_batch, 1, dtype=torch.bool, device=self.device), xm], dim=1)

        inv_loss = self.inverse_loss(h_action, a, a_mask)
        density_loss = self.density_loss(h_density, y_density)
        regularization = self.regularization(z, z_[:, :n_pos])
        return inv_loss, density_loss, regularization

    def _flatten(self, x):
        return torch.cat([x[k] for k in self.order], dim=1)


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
