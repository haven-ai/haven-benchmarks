import os

import torch.optim
from torch import nn
from torch.nn import functional as F
import tqdm
import numpy as np


def get_model(exp_dict):
    model = None
    if exp_dict["model"]["name"] == "mlp":
        model = Mlp(n_classes=10, dropout=False)
    else:
        raise ValueError

    if exp_dict["optimizer"] == "adam":
        model.opt = torch.optim.Adam(
            model.parameters(), lr=exp_dict["lr"], betas=(0.99, 0.999), weight_decay=0.0005)

    elif exp_dict["optimizer"] == "sgd":
        model.opt = torch.optim.SGD(
            model.parameters(), lr=exp_dict["lr"])
    else:
        raise ValueError

    return model


# =====================================================
# MLP
class Mlp(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256],
                 n_classes=10,
                 bias=True, dropout=False):
        super().__init__()

        self.opt = None

        self.dropout=dropout
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)

            if self.dropout:
                out = F.dropout(out, p=0.5)

        logits = self.output_layer(out)

        return logits

    def train_on_loader(model, train_loader):
        model.train()

        n_batches = len(train_loader)
        train_meter = Meter()

        pbar = tqdm.tqdm(total=n_batches)
        for batch in train_loader:
            images, _ = batch
            score_dict = model.train_on_batch(batch)
            train_meter.add(score_dict['train_loss'], images.shape[0])

            pbar.set_description("Training. Loss: %.4f" % train_meter.get_avg_score())
            pbar.update(1)

        pbar.close()

        return {'train_loss': train_meter.get_avg_score()}

    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir_images=None, n_images=2):
        # todo: save dir? n_image?
        self.eval()

        n_batches = len(val_loader)
        val_meter = Meter()
        pbar = tqdm.tqdm(total=n_batches)
        # for i, batch in enumerate(tqdm.tqdm(val_loader)):
        for i, batch in enumerate(val_loader):
            images, _ = batch
            score_dict = self.val_on_batch(batch)
            val_meter.add(score_dict['success'], images.shape[0])
            self.val_on_batch(batch)

            pbar.set_description("Validating. %.4f acc" % val_meter.get_avg_score())
            pbar.update(1)

            # if savedir_images and i < n_images:
            #     os.makedirs(savedir_images, exist_ok=True)
            #     self.val_on_batch(batch)
            #
            #     pbar.set_description("Validating. MAE: %.4f" % val_meter.get_avg_score())

        pbar.close()
        val_acc = val_meter.get_avg_score()
        val_dict = {'val_acc': val_acc}
        return val_dict

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images, labels = batch
        logits = self.forward(images)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(logits, labels.view(-1))
        loss.backward()

        self.opt.step()

        return {"train_loss": loss.item()}

    def val_on_batch(self, batch):
        self.eval()
        images, labels = batch
        logits = self.forward(images)
        probs = logits.sigmoid().cpu().numpy()

        classifications = np.argmax(probs,axis=1)
        return {'success': (classifications == labels.numpy()).sum()}

    def get_state_dict(self):
        state_dict = {"model": self.state_dict(),
                      "opt":self.opt.state_dict()}

        return state_dict
        
    def set_state_dict(self, state_dict):
        self.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])


class Meter:
    def __init__(self):
        self.n_sum = 0.
        self.n_counts = 0.

    def add(self, n_sum, n_counts):
        self.n_sum += n_sum
        self.n_counts += n_counts

    def get_avg_score(self):
        return self.n_sum / self.n_counts
