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
            torch.nn.Linear(n_hidden, n_latent)
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
        zf = torch.cat([z[k] for k in self._order], dim=1)
        mask = torch.cat([pad_mask[k].to(self.device) for k in self._order], dim=1)
        z_proj = self.pre_attention(zf)
        z_agg = self.attention(z_proj, src_key_padding_mask=~mask)
        z_agg = (z_agg * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1).unsqueeze(1)
        return z_agg

    def forward(self, x):
        z = self.encode(x)
        z_agg = self.aggregate(z, x["masks"])
        return z_agg

    def inverse_loss(self, z, z_, a):
        a_logits = self.inverse_forward(z, z_)
        loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device))
        return loss

    def density_loss(self, z, z_, z_n):
        n_batch = z.shape[0]
        y_logits = self.density_forward(z, z_, z_n)
        y = torch.cat([torch.ones(n_batch), torch.zeros(n_batch)], dim=0).to(self.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(y_logits, y)

    def smoothness_loss(self, z, z_):
        mse = torch.nn.functional.mse_loss(z, z_, reduction="none")
        return torch.square(torch.relu(mse-1)).mean()

    def loss(self, x, x_, x_neg, a):
        z_agg = self.forward(x)
        z_agg_ = self.forward(x_)
        zn_agg = self.forward(x_neg)
        inv_loss = self.inverse_loss(z_agg, z_agg_, a)
        density_loss = self.density_loss(z_agg, z_agg_, zn_agg)
        smoothness_loss = self.smoothness_loss(z_agg, z_agg_)
        return inv_loss, density_loss, smoothness_loss
