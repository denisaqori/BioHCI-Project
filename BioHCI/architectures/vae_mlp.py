# based on: https://github.com/pytorch/examples/blob/master/vae/main.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE_MLP(nn.Module):

    def __init__(self):
        super(VAE_MLP, self).__init__()

        self.input_size = 1000  # x.shape[1] * x.shape{2] = 250 * 4

        # encoding
        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # decoding
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, self.input_size)

    def encode(self, x):
        h1 = self.fc1(x)
        h1r = F.relu(h1)

        h21 = self.fc21(h1r)
        h22 = self.fc22(h1r)
        return h21, h22

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.fc3(z)
        h3r = F.relu(h3)
        h4 = self.fc4(h3r)
        o = torch.sigmoid(h4)
        return o

    def forward(self, x):
        x_flat = x.view(-1, x.shape[1] * x.shape[2])
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        sample = self.decode(z)
        return sample, mu, logvar

