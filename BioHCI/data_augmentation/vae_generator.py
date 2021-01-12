# from: https://github.com/pytorch/examples/blob/master/vae/main.py

from __future__ import print_function

from BioHCI.architectures.vae import VAE
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import BioHCI.helpers.utilities as utils
from os.path import join


# should include train, val and test data
class VAE_Generator():
    def __init__(self, train_data_loader, val_data_loader, learning_def, n_epochs=5, log_interval=2,
                 seed=1):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if learning_def.use_cuda else "cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if learning_def.use_cuda else {}
        self.batch_size = learning_def.batch_size

        self.train_loader = train_data_loader
        self.val_loader = val_data_loader
        self.test_loader = None

        self.model = VAE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.n_epochs = n_epochs
        self.log_interval = log_interval

        # should be called independently from other code at different times
        self.perform_cv(self.train_loader, self.val_loader)
        print("")
        # self.test(self.test_loader)



    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[1] * x.shape[2]), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, cat) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            recon_batch, mu, logvar = self.model(data)

            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            data_reshaped = data.view(-1, data.shape[1] * data.shape[2])
            if batch_idx == 4:
                # sample plotting
                sns.set(context='notebook', style='darkgrid', palette='pastel', font='sans-serif', font_scale=1,
                        color_codes=True, rc=None)
                x = np.arange(0, data_reshaped.shape[1])

                data_sample = data_reshaped[100, :].cpu().numpy()
                plt.plot(x, data_sample, label='original data')
                recon_sample = recon_batch[100, :].cpu().detach().numpy()
                plt.plot(x, recon_sample, label='vae-generated')

                plt.legend()
                plt.show()

            # if batch_idx % self.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

    def test(self, test_loader, model_path=None):
        if model_path is None:
            model = self.model.eval()
        else:
            model = torch.load(model_path).eval()

        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(self.batch_size, 250, 4)[:n]])
                    comparison = comparison.cpu()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def perform_cv(self, train_data_loader, val_data_loader):
        for epoch in range(1, self.n_epochs + 1):
            self.train(train_data_loader, epoch)
            self.test(val_data_loader)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(self.device)
                sample = self.model.decode(sample).cpu()
