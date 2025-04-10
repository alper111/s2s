import os

import torch
from tqdm import tqdm
import yaml

from .base import Abstraction


class MarkovStateAbstraction(Abstraction, torch.nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        folder = None
        ckpt_path = None
        if isinstance(config, str):
            folder = config
            config_path = os.path.join(config, "config.yaml")
            ckpt_path = os.path.join(config, "msa.pt")
            config = yaml.safe_load(open(config_path, "r"))
        self.config = config
        self._initialize_layers()
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self.load(folder)
        self.to(config["device"])

    @property
    def order(self):
        return self._order

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def fit(self, train_loader, val_loader, save_path, load_path=None):
        print(self)
        print(f"Number of parameters: {self.n_parameters:,}")
        if load_path is not None:
            self.load(load_path)

        optimizer = torch.optim.Adam([{"params": self.enc_proj.parameters()},
                                      {"params": self.encoder.parameters()},
                                      {"params": self.pre_attention.parameters()},
                                      {"params": self.attention.parameters()},
                                      {"params": self.context.parameters()},
                                      {"params": self.inverse_fc.parameters()},
                                      {"params": self.density_fc.parameters()},
                                      {"params": self.decoder.parameters()},
                                      {"params": self.dec_proj.parameters()}], lr=self.config["lr"],
                                     weight_decay=1e-5)
        mi_optim = torch.optim.Adam(self.mi.parameters(), lr=self.config["lr"], weight_decay=1e-5)
        for e in range(self.config["epoch"]):
            train_results = self._one_iteration(train_loader, self.config["negative_rate"], optimizer, mi_optim)
            train_inv, train_density, _, train_smt, train_recon = train_results
            print(f"Epoch {e + 1}/{self.config['epoch']}, "
                  f"inverse={train_inv:.5f}, "
                  f"density={train_density:.5f}, "
                  f"smoothness={train_smt:.5f}, "
                  f"reconstruction={train_recon:.5f}")
            if val_loader is not None:
                val_results = self._one_iteration(val_loader, self.config["negative_rate"])
                val_inv, val_density, _, val_smt, val_recon = val_results
                print(f"validation: "
                      f"inverse={val_inv:.5f}, "
                      f"density={val_density:.5f}, "
                      f"smoothness={val_smt:.5f}, "
                      f"reconstruction={val_recon:.5f}")

            if (e+1) % self.config["save_freq"] == 0:
                self.save(save_path)
    
    def _one_iteration(self, loader, negative_rate=10, optimizer=None, mi_optim=None):
        avg_inv_loss = 0
        avg_density_loss = 0
        avg_mi_loss = 0
        avg_smoothness_loss = 0
        avg_recon_loss = 0
        for x, a, x_ in tqdm(loader):
            # train the mutual information neural estimator
            for _ in range(1):
                with torch.no_grad():
                    z = self.forward(x)
                mi_loss = self.mi_loss(z)
                if mi_optim is not None:
                    mi_optim.zero_grad()
                    mi_loss.backward()
                    mi_optim.step()
   
            # train the main model
            n = x[self._order[0]].shape[0] * negative_rate
            x_n, _, _ = loader.dataset.sample(n)
            inv_loss, density_loss, mi_loss, smoothness_loss, recon_loss = self.loss(x, x_, x_n, a)
            loss = inv_loss + density_loss + self.config["smoothness_coeff"]*smoothness_loss + \
                self.config["mi_coeff"]*mi_loss + recon_loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_inv_loss += inv_loss.item()
            avg_density_loss += density_loss.item()
            avg_mi_loss += mi_loss.item()
            avg_smoothness_loss += smoothness_loss.item()
            avg_recon_loss += recon_loss.item()
        return (avg_inv_loss / len(loader),
                avg_density_loss / len(loader),
                avg_mi_loss / len(loader),
                avg_smoothness_loss / len(loader),
                avg_recon_loss / len(loader))

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

    def attn(self, z, m):
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

    def pad_and_cat(self, z, zn, z_neg, xm, xm_, xm_neg, a=None):
        n_batch, n_pos, n_dim = z.shape
        n_neg = z_neg.shape[1]

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
            if a is not None:
                a_pad = torch.zeros(n_batch, -n_remaining, a.shape[-1], dtype=torch.float, device=a.device)
                a = torch.cat([a, a_pad], dim=1)
        z_all = torch.cat([z, zn, z_neg], dim=0)
        m_all = torch.cat([xm, xm_, xm_neg], dim=0)
        if a is not None:
            return z_all, m_all, a
        return z_all, m_all

    def inverse_loss(self, h, a):
        a_logits = self.inverse_fc(h)
        if self._cls_type == "softmax":
            assert a.ndim == 2
            a_logits = a_logits.permute(0, 2, 1)
            loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device), reduction="none")
            loss = loss.sum(dim=1).mean()
        elif self._cls_type == "sigmoid":
            assert a.ndim == 3
            loss = torch.nn.functional.binary_cross_entropy_with_logits(a_logits, a.float().to(self.device), reduction="none")
            loss = loss.sum(dim=[1, 2]).mean()
        else:
            raise ValueError(f"Unknown action classification type: {self._cls_type}")
        return loss

    def density_loss(self, h, hn, h_neg, temperature=0.1) -> torch.Tensor:
        n_batch = h.shape[0]
        k = h_neg.shape[0] // n_batch
        h_rep = h.repeat_interleave(k, 0)

        pos_inp = torch.cat([h, hn], dim=1)
        neg_inp = torch.cat([h_rep, h_neg], dim=1)

        pos_logits = self.density_fc(pos_inp)
        neg_logits = self.density_fc(neg_inp).reshape(n_batch, k)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
        y = torch.zeros(n_batch, dtype=torch.long, device=self.device)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss

    def smoothness_loss(self, h, hn, d=1.0):
        mse = torch.nn.functional.mse_loss(h, hn, reduction="none")
        loss = torch.relu(mse - d).square().mean()
        return loss

    def mi_loss(self, z):
        k = 10
        z = z.reshape(-1, z.shape[-1])
        n_batch, n_dim = z.shape
        indices = torch.stack([torch.randperm(n_dim)[:2] for _ in range(n_batch)])
        indices = indices.to(self.device)
        m_indices = indices[:, 1].repeat_interleave(k, 0)
        z1 = torch.stack([indices[:, 0],
                          z[torch.arange(n_batch, device=self.device),
                            indices[:, 0]]], dim=1)
        z2 = torch.stack([indices[:, 1],
                          z[torch.arange(n_batch, device=self.device),
                            indices[:, 1]]], dim=1)
        zj = torch.stack([m_indices,
                          z[torch.randint(0, n_batch, (k*n_batch,), device=self.device),
                            m_indices]], dim=1)
        z_joint = torch.cat([z1, z2], dim=1)
        z_marg = torch.cat([z1.repeat_interleave(k, 0), zj], dim=1)
        pos_logits = self.mi(z_joint)
        neg_logits = self.mi(z_marg).reshape(n_batch, k)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        y = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        return torch.nn.functional.cross_entropy(logits, y)

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
        n_batch, n_pos, _ = z.shape

        xm = self._flatten(x["masks"]).to(self.device)
        xm_ = self._flatten(x_["masks"]).to(self.device)
        xm_neg = self._flatten(x_neg["masks"]).to(self.device)

        # if the positive and negative samples don't have the same
        # number of objects in them...
        z_all, m_all, a_all = self.pad_and_cat(z, zn, z_neg, xm, xm_, xm_neg, a)

        # do the attention computation in one go
        h_all = self.attn(z_all, m_all)
        # now split the attention outputs to state, next_state, and negative samples
        h = h_all[:n_batch, 0]  # (n_batch, n_dim)
        hn = h_all[n_batch:(2*n_batch), 0]  # (n_batch, n_dim)
        h_neg = h_all[(2*n_batch):, 0]  # (n_batch*k, n_dim) where k is the negative rate

        density_loss = self.density_loss(h, hn, h_neg)

        h_pos = torch.cat([h, hn], dim=-1)

        # repeat the aggregate hidden for every object
        h_action = h_pos.repeat_interleave(n_pos+1, 0)
        # the aggregate context vector
        ctx_agg = self.context(torch.full((n_batch, 1), 0, dtype=torch.long, device=self.device))
        ctx_z = self.context(torch.full((n_batch, n_pos), 1, dtype=torch.long, device=self.device))
        z_proj = self.pre_attention(z) + ctx_z
        # contains the aggregate context vector and the projected z
        z_action = torch.cat([ctx_agg, z_proj], dim=1).reshape(n_batch*(n_pos+1), -1)
        # concatenate the processed hidden aggregate with each object's z
        h_action = torch.cat([h_action, z_action], dim=-1).reshape(n_batch, n_pos+1, -1)
        inv_loss = self.inverse_loss(h_action, a_all)
        smoothness_loss = self.smoothness_loss(h, hn)
        mi_loss = -self.mi_loss(z)

        # decoding is a separate process.
        # don't backpropagate the recon loss to the encoder
        n_remaining = z.shape[1] - z_neg.shape[1]
        if n_remaining < 0:
            z = z[:, :(n_pos+n_remaining)]
        x_bar = self.decode(z.detach(), [x[k].shape[1] for k in self.order])
        reconstruction_loss = self.reconstruction_loss(x_bar, x)

        return inv_loss, density_loss, mi_loss, smoothness_loss, reconstruction_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        ckpt_name = os.path.join(path, "msa.pt")
        config_name = os.path.join(path, "config.yaml")
        torch.save(self.state_dict(), ckpt_name)
        yaml.safe_dump(self.config, open(config_name, "w"))

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
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_latent"]))
            else:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]))
                enc_layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*enc_layers)

        self.pre_attention = torch.nn.Sequential(
            torch.nn.Linear(self.config["n_latent"], self.config["n_hidden"]),
            torch.nn.ReLU()
        )
        _att_layers = torch.nn.TransformerEncoderLayer(d_model=self.config["n_hidden"], nhead=4, batch_first=True)
        self.attention = torch.nn.TransformerEncoder(_att_layers, num_layers=2)
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

        # for mi estimation
        self.mi = torch.nn.Sequential(
            torch.nn.Linear(4, self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], 1)
        )

    def _flatten(self, x):
        return torch.cat([x[k] for k in self.order], dim=1)


class MSAFlat(Abstraction, torch.nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        folder = None
        ckpt_path = None
        if isinstance(config, str):
            folder = config
            config_path = os.path.join(config, "config.yaml")
            ckpt_path = os.path.join(config, "msa.pt")
            config = yaml.safe_load(open(config_path, "r"))
        self.config = config
        self._initialize_layers()
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self.load(folder)
        self.to(config["device"])

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def fit(self, train_loader, val_loader, config, save_path, load_path=None):
        print(self)
        print(f"Number of parameters: {self.n_parameters:,}")
        if load_path is not None:
            self.load(load_path)

        optimizer = torch.optim.Adam([{"params": self.encoder.parameters()},
                                      {"params": self.inverse_fc.parameters()},
                                      {"params": self.density_fc.parameters()},
                                      {"params": self.decoder.parameters()}], lr=config["lr"],
                                     weight_decay=1e-5)
        mi_optim = torch.optim.Adam(self.mi.parameters(), lr=config["lr"], weight_decay=1e-5)
        for e in range(config["epoch"]):
            train_results = self._one_iteration(train_loader, config["negative_rate"], optimizer, mi_optim)
            val_results = self._one_iteration(val_loader, config["negative_rate"])
            (train_inv, train_density, _, train_smt, train_recon) = train_results
            (val_inv, val_density, _, val_smt, val_recon) = val_results

            print(f"Epoch {e + 1}/{config['epoch']}, "
                  f"inverse={train_inv:.5f} ({val_inv:.5f}), "
                  f"density={train_density:.5f} ({val_density:.5f}), "
                  f"smoothness={train_smt:.5f} ({val_smt:.5f}), "
                  f"reconstruction={train_recon:.5f} ({val_recon:.5f})")

            if (e+1) % config["save_freq"] == 0:
                self.save(save_path)

    def _one_iteration(self, loader, negative_rate=10, optimizer=None, mi_optim=None):
        avg_inv_loss = 0
        avg_density_loss = 0
        avg_mi_loss = 0
        avg_smoothness_loss = 0
        avg_recon_loss = 0
        for x, a, x_ in tqdm(loader):
            # train the mutual information neural estimator
            for _ in range(1):
                with torch.no_grad():
                    z = self.forward(x)
                mi_loss = self.mi_loss(z)
                if mi_optim is not None:
                    mi_optim.zero_grad()
                    mi_loss.backward()
                    mi_optim.step()

            # train the main model
            n = x.shape[0] * negative_rate
            x_n, _, _ = loader.dataset.sample(n)
            inv_loss, density_loss, mi_loss, smoothness_loss, recon_loss = self.loss(x, x_, x_n, a)
            loss = inv_loss + density_loss + self.config["smoothness_coeff"]*smoothness_loss + \
                self.config["mi_coeff"]*mi_loss + recon_loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_inv_loss += inv_loss.item()
            avg_density_loss += density_loss.item()
            avg_mi_loss += mi_loss.item()
            avg_smoothness_loss += smoothness_loss.item()
            avg_recon_loss += recon_loss.item()
        return (avg_inv_loss / len(loader),
                avg_density_loss / len(loader),
                avg_mi_loss / len(loader),
                avg_smoothness_loss / len(loader),
                avg_recon_loss / len(loader))

    def encode(self, x):
        return self.encoder(x.to(self.device))

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
        a_logits = self.inverse_fc(z_cat)
        loss = torch.nn.functional.cross_entropy(a_logits, a.to(self.device))
        return loss

    def density_loss(self, z, zn, z_neg, temperature=0.1) -> torch.Tensor:
        n_batch = z.shape[0]
        k = z_neg.shape[0] // n_batch
        z_rep = z.repeat_interleave(k, 0)

        pos_inp = torch.cat([z, zn], dim=1)
        neg_inp = torch.cat([z_rep, z_neg], dim=1)

        pos_logits = self.density_fc(pos_inp)
        neg_logits = self.density_fc(neg_inp).reshape(n_batch, k)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
        y = torch.zeros(n_batch, dtype=torch.long, device=self.device)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss

    def smoothness_loss(self, z, zn, d=1.0):
        mse = torch.nn.functional.mse_loss(z, zn, reduction="none")
        loss = torch.relu(mse - d).square().mean()
        return loss

    def mi_loss(self, z):
        k = 10
        n_batch, n_dim = z.shape
        indices = torch.stack([torch.randperm(n_dim)[:2] for _ in range(n_batch)])
        indices = indices.to(self.device)
        m_indices = indices[:, 1].repeat_interleave(k, 0)
        z1 = torch.stack([indices[:, 0],
                          z[torch.arange(n_batch, device=self.device),
                            indices[:, 0]]], dim=1)
        z2 = torch.stack([indices[:, 1],
                          z[torch.arange(n_batch, device=self.device),
                            indices[:, 1]]], dim=1)
        zj = torch.stack([m_indices,
                          z[torch.randint(0, n_batch, (k*n_batch,), device=self.device),
                            m_indices]], dim=1)
        z_joint = torch.cat([z1, z2], dim=1)
        z_marg = torch.cat([z1.repeat_interleave(k, 0), zj], dim=1)
        pos_logits = self.mi(z_joint)
        neg_logits = self.mi(z_marg).reshape(n_batch, k)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        y = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        return torch.nn.functional.cross_entropy(logits, y)

    def reconstruction_loss(self, x_pred, x_true):
        return torch.nn.functional.mse_loss(x_pred, x_true.to(self.device))

    def loss(self, x, x_, x_neg, a):
        z = self.forward(x)
        zn = self.forward(x_)
        z_neg = self.forward(x_neg)

        inv_loss = self.inverse_loss(z, zn, a)
        density_loss = self.density_loss(z, zn, z_neg)
        smoothness_loss = self.smoothness_loss(z, zn)
        mi_loss = -self.mi_loss(z)

        # decoding is a separate process.
        # don't backpropagate the recon loss to the encoder
        x_bar = self.decode(z.detach())
        reconstruction_loss = self.reconstruction_loss(x_bar, x)
        return inv_loss, density_loss, mi_loss, smoothness_loss, reconstruction_loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        ckpt_name = os.path.join(path, "msa.pt")
        config_name = os.path.join(path, "config.yaml")
        torch.save(self.state_dict(), ckpt_name)
        yaml.safe_dump(self.config, open(config_name, "w"))

    def load(self, path):
        name = os.path.join(path, "msa.pt")
        self.load_state_dict(torch.load(name, map_location=self.device, weights_only=True))

    def _initialize_layers(self):
        assert len(self.config["input_dims"]) == 1
        in_dim = self.config["input_dims"][0][1]

        # encoder
        enc_layers = [
            torch.nn.Linear(in_dim, self.config["n_hidden"]),
            torch.nn.ReLU(),
        ]
        for i in range(self.config["n_layers"]):
            if i == self.config["n_layers"] - 1:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_latent"]))
            else:
                enc_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]))
                enc_layers.append(torch.nn.ReLU())
        self.encoder = torch.nn.Sequential(*enc_layers)

        self.inverse_fc = torch.nn.Sequential(
            torch.nn.Linear(2*self.config["n_latent"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["action_dim"])
        )

        self.density_fc = torch.nn.Sequential(
            torch.nn.Linear(2*self.config["n_latent"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]),
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
            dec_layers.append(torch.nn.ReLU())
        dec_layers.append(torch.nn.Linear(self.config["n_hidden"], self.config["input_dims"][0][1]))
        self.decoder = torch.nn.Sequential(*dec_layers)

        # for mi estimation
        self.mi = torch.nn.Sequential(
            torch.nn.Linear(4, self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], self.config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config["n_hidden"], 1)
        )

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
