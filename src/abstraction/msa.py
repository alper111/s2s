import os

import torch

from .base import Abstraction


class MarkovStateAbstraction(Abstraction, torch.nn.Module):
    def __init__(self, config):
        Abstraction.__init__(self, config)
        torch.nn.Module.__init__(self)
        self._initialize_layers()

    @property
    def order(self):
        return self._order

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def fit(self, loader, config, save_path, load_path=None):
        if load_path is not None:
            self.load(load_path)

        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        for e in range(config["epoch"]):
            avg_inv_loss = 0
            avg_density_loss = 0
            avg_reg_loss = 0
            avg_recon_loss = 0
            for x, a, x_ in loader:
                n = x["objects"].shape[0] * config["negative_rate"]
                x_n, _, _ = loader.dataset.sample(n)
                inv_loss, density_loss, reg_loss, recon_loss = self.loss(x, x_, x_n, a)
                loss = inv_loss + density_loss + config["beta"]*reg_loss + recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_inv_loss += inv_loss.item()
                avg_density_loss += density_loss.item()
                avg_reg_loss += reg_loss.item()
                avg_recon_loss += recon_loss.item()
            print(f"Epoch {e + 1}/{config['epoch']}, "
                  f"inverse={avg_inv_loss / len(loader):.5f}, "
                  f"density={avg_density_loss / len(loader):.5f}, "
                  f"reg={avg_reg_loss / len(loader):.5f}, "
                  f"recon={avg_recon_loss / len(loader):.5f}")

            if (e+1) % config["save_freq"] == 0:
                self.save(save_path)

    def encode(self, x, return_gating=False):
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
            proj_i = self.enc_proj[mod_i](inputs)
            projs.append(proj_i)
            mod_tokens.append(tokens)
        projs = torch.cat(projs, dim=1)
        feats, g = self.encoder(projs)
        outputs = []
        gatings = []
        it = 0
        for tokens in mod_tokens:
            mod_outs = []
            mod_g = []
            for t_i in tokens:
                mod_outs.append(feats[:, it:(it+t_i)])
                mod_g.append(g[:, it:(it+t_i)])
                it += t_i
            outputs.append(mod_outs)
            gatings.append(mod_g)

        return_feats = []
        return_gatings = []
        for i in range(n):
            f = []
            g = []
            for out, gate in zip(outputs, gatings):
                f.append(out[i])
                g.append(gate[i])
            f = torch.cat(f, dim=1)
            g = torch.cat(g, dim=1)
            return_feats.append(f)
            return_gatings.append(g)
        if n == 1:
            return_feats = return_feats[0]
            return_gatings = return_gatings[0]
        if return_gating:
            return return_feats, return_gatings
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

    def inverse_forward(self, h):
        return self.inverse_fc(h)

    def density_forward(self, h):
        return self.density_fc(h)

    def decode(self, z, tokens):
        # make sure the encodings are detached
        # as we don't want to backpropagate any
        # gradients through the decoder
        z = z.detach()
        h = self.decoder(z)
        h = h.split(tokens, dim=1)
        outs = {}
        for h_i, proj_i in zip(h, self.order):
            if h_i.numel() == 0:
                continue
            out_i = self.dec_proj[proj_i](h_i)
            outs[proj_i] = out_i
        return outs

    def forward(self, x, return_gating=False):
        return self.encode(x, return_gating=return_gating)

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

    def regularization(self, g):
        g = g.flatten(0, -2)
        g_avg = g.mean(dim=0)
        entropy = (g_avg*torch.log(g_avg+1e-12) + (1-g_avg)*torch.log(1-g_avg+1e-12))
        y = torch.zeros_like(g)
        l1_loss = torch.nn.functional.l1_loss(g, y, reduction="none")
        loss = entropy.sum() + l1_loss.sum(dim=-1).mean()
        return loss

    def reconstruction_loss(self, x_pred, x_true):
        loss = 0.0
        for k in self.order:
            loss_k = torch.nn.functional.mse_loss(x_pred[k], x_true[k].to(self.device), reduction="none")
            loss_k = loss_k.mean(axis=-1)
            mask_k = x_true["masks"][k].clone().to(self.device)
            loss_k = (loss_k * mask_k).sum() / mask_k.sum()
            loss += loss_k
        return loss

    def loss(self, x, x_, x_neg, a):
        (z, zn), (g_z, _) = self.forward([x, x_], return_gating=True)
        z_neg = self.forward(x_neg)
        n_batch, n_pos, n_dim = z.shape
        n_neg = z_neg.shape[1]

        xm = self._flatten(x["masks"]).to(self.device)
        xm_ = self._flatten(x_["masks"]).to(self.device)
        xm_neg = self._flatten(x_neg["masks"]).to(self.device)

        rep_count = (z.shape[0] + z_neg.shape[0]) // z.shape[0]
        z_init = z.repeat(rep_count, 1, 1)
        m_init = xm.repeat(rep_count, 1)

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
            zn = torch.cat([zn, zeros], dim=1)
            mask_zeros = torch.zeros(n_batch, -n_remaining, dtype=torch.bool, device=self.device)
            xm_ = torch.cat([xm_, mask_zeros], dim=1)
        z_next = torch.cat([zn, z_neg], dim=0)
        m_next = torch.cat([xm_, xm_neg], dim=0)

        h_all = self.attn_forward(z_init, z_next, m_init, m_next)
        h_density = h_all[:, 0]  # for g(y | z, z') estimation
        y_density = torch.cat([torch.ones(n_batch), torch.zeros(z_neg.shape[0])], dim=0).to(self.device)
        h_action = h_all[:n_batch, 1:(n_pos+2)]
        a_mask = torch.cat([torch.ones(n_batch, 1, dtype=torch.bool, device=self.device), xm], dim=1)

        inv_loss = self.inverse_loss(h_action, a, a_mask)
        density_loss = self.density_loss(h_density, y_density)
        regularization = self.regularization(g_z)

        # decoding is a separate process.
        # don't backpropagate the recon loss to the encoder
        x_bar = self.decode(z.detach(), [x[k].shape[1] for k in self.order])
        reconstruction_loss = self.reconstruction_loss(x_bar, x)

        return inv_loss, density_loss, regularization, reconstruction_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        name = os.path.join(path, "msa.pt")
        torch.save(self.state_dict(), name)

    def load(self, path):
        name = os.path.join(path, "msa.pt")
        self.load_state_dict(torch.load(name, map_location=self.device, weights_only=True))

    def _initialize_layers(self):
        self._order = [x[0] for x in self.config["input_dims"]]
        self._cls_type = self.config["action_classification_type"]

        # encoder projections
        self.enc_proj = torch.nn.ModuleDict(
            {key: torch.nn.Sequential(
                torch.nn.Linear(value, self.config["n_hidden"]),
                torch.nn.ReLU())
             for (key, value) in self.config["input_dims"]})

        # encoder
        enc_layers = []
        for i in range(self.config["n_layers"]):
            if i == self.config["n_layers"] - 1:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_latent"]*2))
                enc_layers.append(GumbelGLU(hard=True))
            else:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]))
                enc_layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*enc_layers)

        self.pre_attention = torch.nn.Sequential(
            torch.nn.Linear(self.config["n_latent"], self.config["n_hidden"]),
            torch.nn.ReLU()
        )
        _att_layers = torch.nn.TransformerEncoderLayer(d_model=self.config["n_hidden"], nhead=4, batch_first=True)
        self.attention = torch.nn.TransformerEncoder(_att_layers, num_layers=4)
        self.context = torch.nn.Embedding(4, self.config["n_hidden"])

        self.inverse_fc = torch.nn.Sequential(
            torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["action_dim"])
        )
        self.density_fc = torch.nn.Sequential(
            torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], 1)
        )

        # decoder is only for visualization purposes,
        # has nothing to do with the training
        dec_layers = []
        for i in range(self.config["n_layers"]):
            if i == 0:
                in_dim = self.config["n_latent"]
            else:
                in_dim = self.config["n_hidden"]
            dec_layers.append(torch.nn.Linear(in_dim, self.config["n_hidden"]))
            dec_layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*dec_layers)

        # decoder projections
        self.dec_proj = torch.nn.ModuleDict(
            {key: torch.nn.Linear(self.config["n_hidden"], value)
             for (key, value) in self.config["input_dims"]})

    def _flatten(self, x):
        return torch.cat([x[k] for k in self.order], dim=1)


class GumbelGLU(torch.nn.Module):
    def __init__(self, hard=False, T=1.0):
        super(GumbelGLU, self).__init__()
        self.hard = hard
        self.T = T

    def forward(self, x):
        if not self.training:
            return torch.nn.functional.glu(x)
        else:
            n_dim = x.shape[-1] // 2
            g = gumbel_sigmoid(x[..., :n_dim], self.T, self.hard)
            return g * x[..., n_dim:], g


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
