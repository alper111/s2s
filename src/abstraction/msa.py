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
            torch.nn.Linear(n_latent * 2, n_hidden),
            torch.nn.ReLU()
        )
        _inv_att_layers = torch.nn.TransformerEncoderLayer(d_model=n_hidden, nhead=4, batch_first=True)
        _den_att_layers = torch.nn.TransformerEncoderLayer(d_model=n_hidden, nhead=4, batch_first=True)
        # fix num_layers to 4 for now
        self.inverse_att = torch.nn.TransformerEncoder(_inv_att_layers, num_layers=4)
        self.inverse_fc = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, action_dim)
        )
        self.density_att = torch.nn.TransformerEncoder(_den_att_layers, num_layers=4)
        self.density_fc = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        return {k: self.feature(self.projection[k](x[k].to(self.device))) for k in x if k != "masks"}

    def inverse_loss(self, z, z_, a, pad_mask):
        zf = torch.cat([z[k] for k in self._order], dim=1)
        zf_ = torch.cat([z_[k] for k in self._order], dim=1)
        z_all = torch.cat([zf, zf_], dim=-1)
        mask = torch.cat([pad_mask[k].to(self.device) for k in self._order], dim=1)

        z_proj = self.pre_attention(z_all)
        agg = self.inverse_att(z_proj, src_key_padding_mask=~mask)
        a_logits = self.inverse_fc(agg)
        return torch.nn.functional.binary_cross_entropy_with_logits(a_logits, a.to(self.device))

    def density_loss(self, z, z_, pad_mask):
        zf = torch.cat([z[k] for k in self._order], dim=1)
        zf_ = torch.cat([z_[k] for k in self._order], dim=1)
        mask = torch.cat([pad_mask[k].to(self.device) for k in self._order], dim=1)

        z_pos = torch.cat([zf, zf_], dim=-1)
        z_neg = torch.cat([zf, zf_[torch.randperm(zf_.shape[0])]], dim=-1)
        z_all = torch.cat([z_pos, z_neg], dim=0)
        mask_all = torch.cat([mask, mask], dim=0)
        y = torch.cat([torch.ones(z_pos.shape[0]), torch.zeros(z_neg.shape[0])], dim=0).to(self.device)

        z_proj = self.pre_attention(z_all)
        agg = self.density_att(z_proj, src_key_padding_mask=~mask_all)
        agg = agg.mean(dim=1)
        y_logits = self.density_fc(agg).squeeze()
        return torch.nn.functional.binary_cross_entropy_with_logits(y_logits, y)

    def smoothness_loss(self, z, z_):
        mse_all = 0.0
        for k in self._order:
            mse = torch.nn.functional.mse_loss(z[k], z_[k], reduction="none")
            mse_all = mse_all + torch.square(torch.relu(mse - 1)).mean()
        return mse_all

    def forward(self, x, x_):
        x_all = {k: torch.cat([x[k], x_[k]], dim=0) for k in x if k != "masks"}
        z_all = self.encode(x_all)
        z = {k: z_all[k][:x[k].shape[0]] * x["masks"][k].to(self.device).unsqueeze(2) for k in x if k != "masks"}
        z_ = {k: z_all[k][x[k].shape[0]:] * x["masks"][k].to(self.device).unsqueeze(2) for k in x if k != "masks"}
        return z, z_

    def loss(self, x, x_, a):
        z, z_ = self.forward(x, x_)
        inv_loss = self.inverse_loss(z, z_, a, x["masks"])
        density_loss = self.density_loss(z, z_, x["masks"])
        smoothness_loss = self.smoothness_loss(z, z_)
        return inv_loss, density_loss, smoothness_loss
