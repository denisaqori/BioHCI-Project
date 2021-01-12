# based on: https://github.com/pytorch/examples/blob/master/vae/main.py

import torch
import torch.nn as nn
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.input_size = 1000  # x.shape[1] * x.shape{2] = 250 * 4

        # encoding
        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # decoding
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h21 = self.fc21(h1)
        h22 = self.fc22(h1)
        return h21, h22

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)
        return torch.sigmoid(h4)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1] * x.shape[2]))
        z = self.reparameterize(mu, logvar)
        sample = self.decode(z)
        return sample, mu, logvar

