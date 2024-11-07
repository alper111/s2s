import os

import torch
from tqdm import tqdm

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
        print(self)
        if load_path is not None:
            self.load(load_path)

        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        for e in range(config["epoch"]):
            avg_inv_loss = 0
            avg_density_loss = 0
            avg_recon_loss = 0
            for x, a, x_ in tqdm(loader):
                n = x[self._order[0]].shape[0] * 10
                x_n, _, _ = loader.dataset.sample(n)
                inv_loss, density_loss, recon_loss = self.loss(x, x_, x_n, a)
                loss = inv_loss + density_loss + recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_inv_loss += inv_loss.item()
                avg_density_loss += density_loss.item()
                avg_recon_loss += recon_loss.item()
            print(f"Epoch {e + 1}/{config['epoch']}, "
                  f"inverse={avg_inv_loss / len(loader):.5f}, "
                  f"density={avg_density_loss / len(loader):.5f}, "
                  f"recon={avg_recon_loss / len(loader):.5f}")

            if (e+1) % config["save_freq"] == 0:
                self.save(save_path)

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
            proj_i = self.enc_proj[mod_i](inputs)
            projs.append(proj_i)
            mod_tokens.append(tokens)
        projs = torch.cat(projs, dim=1)
        feats = self.encoder(projs)
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

    def attn_forward(self, z, m):
        n_batch, z_token, _ = z.shape
        ctx_agg = self.context(torch.full((n_batch, 1), 0, dtype=torch.long, device=self.device))
        ctx_z = self.context(torch.full((n_batch, z_token), 1, dtype=torch.long, device=self.device))
        z = self.pre_attention(z)
        z = z + ctx_z
        inputs = torch.cat([ctx_agg, z], dim=1)
        mask = torch.cat([torch.ones(n_batch, 1, dtype=torch.bool, device=self.device), m], dim=1)
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

    def forward(self, x):
        return self.encode(x)

    def inverse_loss(self, h, a):
        a_logits = self.inverse_forward(h)
        if self._cls_type == "softmax":
            assert a.ndim == 2
            a_logits = a_logits.permute(0, 2, 1)
            loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device), reduction="none")
            loss = loss.sum(dim=1).mean()
        elif self._cls_type == "sigmoid":
            assert a.ndim == 3
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a_logits, a.to(self.device), reduction="none")
            loss = loss.sum(dim=[1, 2]).mean()
        else:
            raise ValueError(f"Unknown action classification type: {self._cls_type}")
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
        z, zn = self.forward([x, x_])
        z_neg = self.forward(x_neg)
        n_batch, n_pos, n_dim = z.shape
        n_neg = z_neg.shape[1]

        xm = self._flatten(x["masks"]).to(self.device)
        xm_ = self._flatten(x_["masks"]).to(self.device)
        xm_neg = self._flatten(x_neg["masks"]).to(self.device)

        # if the positive and negative samples don't have the same
        # number of objects in them...
        n_remaining = n_pos - n_neg
        if n_remaining > 0:
            zeros = torch.zeros(z_neg.shape[0], n_remaining, n_dim, dtype=torch.float, device=self.device)
            z_neg = torch.cat([z_neg, zeros], dim=1)
            mask_zeros = torch.zeros(z_neg.shape[0], n_remaining, dtype=torch.bool, device=self.device)
            xm_neg = torch.cat([xm_neg, mask_zeros], dim=1)
            n_neg = n_pos
        elif n_remaining < 0:
            zeros = torch.zeros((n_batch, -n_remaining, n_dim), dtype=torch.float, device=self.device)
            z = torch.cat([z, zeros], dim=1)
            zn = torch.cat([zn, zeros], dim=1)
            mask_zeros = torch.zeros(n_batch, -n_remaining, dtype=torch.bool, device=self.device)
            xm = torch.cat([xm, mask_zeros], dim=1)
            xm_ = torch.cat([xm_, mask_zeros], dim=1)
            n_pos = n_neg
            a_pad = torch.zeros(n_batch, -n_remaining, a.shape[-1], dtype=torch.float, device=a.device)
            a = torch.cat([a, a_pad], dim=1)
        z_all = torch.cat([z, zn, z_neg], dim=0)
        m_all = torch.cat([xm, xm_, xm_neg], dim=0)

        h_all = self.attn_forward(z_all, m_all)
        h = h_all[:n_batch, 0]  # (n_batch, n_dim)
        hn = h_all[n_batch:(2*n_batch), 0]  # (n_batch, n_dim)
        h_neg = h_all[(2*n_batch):, 0]  # (x_neg.shape[0], n_dim)

        h_pos = torch.cat([h, hn], dim=-1)
        rep_count = h_neg.shape[0] // n_batch
        if rep_count > 1:
            h = h.repeat(rep_count, 1)
        h_neg = torch.cat([h, h_neg], dim=-1)
        h_density = torch.cat([h_pos, h_neg], dim=0)
        y_density = torch.cat([torch.ones(n_batch), torch.zeros(z_neg.shape[0])], dim=0).to(self.device)

        h_action = h_pos.repeat_interleave(n_pos+1, 0)
        z_agg = self.context(torch.full((n_batch, 1), 0, dtype=torch.long, device=self.device))
        z_proj = self.pre_attention(z)
        z_action = torch.cat([z_agg, z_proj], dim=1).reshape(n_batch*(n_pos+1), -1)
        h_action = torch.cat([h_action, z_action], dim=-1).reshape(n_batch, n_pos+1, -1)

        inv_loss = self.inverse_loss(h_action, a)
        density_loss = self.density_loss(h_density, y_density)

        # decoding is a separate process.
        # don't backpropagate the recon loss to the encoder
        if n_remaining < 0:
            z = z[:, :(n_pos+n_remaining)]
        x_bar = self.decode(z.detach(), [x[k].shape[1] for k in self.order])
        reconstruction_loss = self.reconstruction_loss(x_bar, x)

        return inv_loss, density_loss, reconstruction_loss

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
                torch.nn.LayerNorm(self.config["n_hidden"]),
                torch.nn.ReLU())
             for (key, value) in self.config["input_dims"]})

        # encoder
        enc_layers = []
        for i in range(self.config["n_layers"]):
            if i == self.config["n_layers"] - 1:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_latent"]))
                enc_layers.append(torch.nn.LayerNorm(self.config["n_latent"]))
            else:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]))
                enc_layers.append(torch.nn.LayerNorm(self.config["n_hidden"]))
                enc_layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*enc_layers)

        self.pre_attention = torch.nn.Sequential(
            torch.nn.Linear(self.config["n_latent"], self.config["n_hidden"]),
            torch.nn.ReLU()
        )
        _att_layers = torch.nn.TransformerEncoderLayer(d_model=self.config["n_hidden"], nhead=4, batch_first=True)
        self.attention = torch.nn.TransformerEncoder(_att_layers, num_layers=4)
        self.context = torch.nn.Embedding(2, self.config["n_hidden"])

        self.inverse_fc = torch.nn.Sequential(
            torch.nn.Linear(3*self.config["n_hidden"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["action_dim"])
        )
        self.density_fc = torch.nn.Sequential(
            torch.nn.Linear(2*self.config["n_hidden"], self.config["n_hidden"]),
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


class MSAFlat(Abstraction, torch.nn.Module):
    def __init__(self, config):
        Abstraction.__init__(self, config)
        torch.nn.Module.__init__(self)
        self._initialize_layers()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def fit(self, loader, config, save_path, load_path=None):
        print(self)
        if load_path is not None:
            self.load(load_path)

        optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        for e in range(config["epoch"]):
            avg_inv_loss = 0
            avg_density_loss = 0
            avg_recon_loss = 0
            for x, a, x_ in tqdm(loader):
                n = x.shape[0] * 10
                x_n, _, _ = loader.dataset.sample(n)
                inv_loss, density_loss, recon_loss = self.loss(x, x_, x_n, a)
                loss = inv_loss + density_loss + recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_inv_loss += inv_loss.item()
                avg_density_loss += density_loss.item()
                avg_recon_loss += recon_loss.item()
            print(f"Epoch {e + 1}/{config['epoch']}, "
                  f"inverse={avg_inv_loss / len(loader):.5f}, "
                  f"density={avg_density_loss / len(loader):.5f}, "
                  f"recon={avg_recon_loss / len(loader):.5f}")

            if (e+1) % config["save_freq"] == 0:
                self.save(save_path)

    def encode(self, x):
        return self.encoder(x)

    def inverse_forward(self, h):
        return self.inverse_fc(h)

    def density_forward(self, h):
        return self.density_fc(h)

    def decode(self, z):
        # make sure the encodings are detached
        # as we don't want to backpropagate any
        # gradients through the decoder
        z = z.detach()
        x_bar = self.decoder(z)
        return x_bar

    def forward(self, x):
        return self.encode(x.to(self.device))

    def inverse_loss(self, z, zn, a):
        z_cat = torch.cat([z, zn], dim=-1)
        a_logits = self.inverse_forward(z_cat)
        loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device))
        return loss

    def density_loss(self, z, zn, z_neg):
        z_pos = torch.cat([z, zn], dim=-1)
        rep_count = z_neg.shape[0] // z.shape[0]
        z_rep = z.repeat_interleave(rep_count, 0)
        z_neg = torch.cat([z_rep, z_neg], dim=-1)
        pos_logits = self.density_forward(z_pos).flatten()
        neg_logits = self.density_forward(z_neg).flatten()
        y = torch.cat([torch.ones(pos_logits.shape[0]), torch.zeros(neg_logits.shape[0])], dim=0).to(self.device)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        return loss

    def regularization(self, g):
        g = g.flatten(0, -2)
        g_avg = g.mean(dim=0)
        entropy = (g_avg*torch.log(g_avg+1e-12) + (1-g_avg)*torch.log(1-g_avg+1e-12))
        y = torch.zeros_like(g)
        l1_loss = torch.nn.functional.l1_loss(g, y, reduction="none")
        loss = entropy.sum() + l1_loss.sum(dim=-1).mean()
        return loss

    def reconstruction_loss(self, x_pred, x_true):
        return torch.nn.functional.mse_loss(x_pred, x_true.to(self.device))

    def loss(self, x, x_, x_neg, a):
        z = self.forward(x)
        zn = self.forward(x_)
        z_neg = self.forward(x_neg)

        inv_loss = self.inverse_loss(z, zn, a)
        density_loss = self.density_loss(z, zn, z_neg)

        # decoding is a separate process.
        # don't backpropagate the recon loss to the encoder
        x_bar = self.decode(z.detach())
        reconstruction_loss = self.reconstruction_loss(x_bar, x)
        return inv_loss, density_loss, reconstruction_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        name = os.path.join(path, "msa.pt")
        torch.save(self.state_dict(), name)

    def load(self, path):
        name = os.path.join(path, "msa.pt")
        self.load_state_dict(torch.load(name, map_location=self.device, weights_only=True))

    def _initialize_layers(self):
        assert len(self.config["input_dims"]) == 1
        in_dim = self.config["input_dims"][0][1]

        # encoder
        enc_layers = [
            torch.nn.Linear(in_dim, self.config["n_hidden"]),
            torch.nn.LayerNorm(self.config["n_hidden"]),
            torch.nn.ReLU(),
        ]
        for i in range(self.config["n_layers"]):
            if i == self.config["n_layers"] - 1:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_latent"]))
                enc_layers.append(torch.nn.LayerNorm(self.config["n_latent"]))
            else:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]))
                enc_layers.append(torch.nn.LayerNorm(self.config["n_hidden"]))
                enc_layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*enc_layers)

        self.inverse_fc = torch.nn.Sequential(
            torch.nn.Linear(2*self.config["n_latent"], self.config["n_hidden"]),
            torch.nn.LayerNorm(self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["action_dim"])
        )
        self.density_fc = torch.nn.Sequential(
            torch.nn.Linear(2*self.config["n_latent"], self.config["n_hidden"]),
            torch.nn.LayerNorm(self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], 1)
        )

        # decoder is only for visualization purposes,
        # has nothing to do with training abstractions
        dec_layers = []
        for i in range(self.config["n_layers"]):
            if i == 0:
                in_dim = self.config["n_latent"]
            else:
                in_dim = self.config["n_hidden"]
            dec_layers.append(torch.nn.Linear(in_dim, self.config["n_hidden"]))
            dec_layers.append(torch.nn.LayerNorm(self.config["n_hidden"]))
            dec_layers.append(torch.nn.ReLU())
        dec_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["input_dims"][0][1]))
        self.decoder = torch.nn.Sequential(*dec_layers)


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
