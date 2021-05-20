# from: https://github.com/pytorch/examples/blob/master/vae/main.py

from __future__ import print_function

import math

from matplotlib import gridspec

from BioHCI.architectures.vae_lstm import VAE_LSTM
from BioHCI.architectures.vae_lstm_attn import VAE_LSTM_ATTN
from BioHCI.architectures.vae_mlp import VAE_MLP
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# should include train, val and test data
from BioHCI.data_processing.keypoint_description.ELD import ELD


class VAE_Generator():
    def __init__(self, train_data_loader, val_data_loader, device, batch_size, n_epochs=200, log_interval=2,
                 name="General"):
        self.batch_size = batch_size

        self.train_loader = train_data_loader
        self.val_loader = val_data_loader
        self.test_loader = None

        self.device = device
        # self.model = VAE_MLP().to(self.device)
        # self.model = VAE_LSTM().to(self.device)
        self.model = VAE_LSTM_ATTN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.name = name

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = self.get_recon_loss(recon_x, x, "SmoothL1")
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Burgess, et. al 2017 (https://arxiv.org/pdf/1804.03599.pdf)
        beta = 3 # beta > 1
        return recon_loss + beta*KLD
        # return smoothL1Loss + KLD

    def get_recon_loss(self, recon_x, x, type="SmoothL1"):
        loss = None
        if type == "MSD":
            recon_x_reshaped = recon_x.reshape(-1, recon_x.shape[1] * recon_x.shape[2])
            x_reshaped = x.reshape(-1, x.shape[1] * x.shape[2])
            # replacing previous reconstruction loss: from binary cross-entropy used for images to MSE for signals
            loss = F.mse_loss(recon_x_reshaped, x_reshaped, size_average=None, reduce=None, reduction='mean')
        elif type == "SmoothL1":
            recon_x_reshaped = recon_x.reshape(-1, recon_x.shape[1] * recon_x.shape[2])
            x_reshaped = x.reshape(-1, x.shape[1] * x.shape[2])
            smoothL1 = nn.SmoothL1Loss()
            loss = smoothL1(recon_x_reshaped, x_reshaped)
        elif type == "ELD":
            loss = ELD.compute_distance(recon_x.cpu().detach().numpy(), x.cpu().detach().numpy())
        else:
            print("Invalid loss function")
            exit()

        return loss

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

            if batch_idx == 0 and epoch % 20 == 0:
                self.plot_sample_values(data, recon_batch, epoch)

            # if batch_idx % self.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item() / len(data)))

        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
        return avg_loss

    @staticmethod
    def plot_sample_values(original_data, recon_data, epoch, batch_sample=0):
        assert original_data.shape == recon_data.shape, "The real and reconstructed dataset need to have the same " \
                                                        "dimensions."
        nplot_cols = 2
        nplot_rows = int(math.ceil(original_data.shape[2]/nplot_cols))

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"Real and reconstructed data: {epoch} epochs", fontsize=18)

        G = gridspec.GridSpec(nrows=nplot_rows, ncols=nplot_cols)

        sns.set(context='notebook', style='darkgrid', palette='pastel', font='sans-serif', font_scale=1,
                color_codes=True, rc=None)

        x = np.arange(0, original_data.shape[1])
        num = 0
        for col in range(0, nplot_cols):
            for row in range(0, nplot_rows):
                if num < original_data.shape[2]:
                    ax = fig.add_subplot(G[row, col])

                    data_sample = original_data[batch_sample, :, num].cpu().numpy()
                    recon_sample = recon_data[batch_sample, :, num].cpu().detach().numpy()

                    ax.plot(x, data_sample, label="original data")
                    ax.plot(x, recon_sample, label="vae-generated")

                    ax.set_ylabel("Voltage Frequency Gains (V)", fontsize=14, labelpad=10)
                    ax.set_xlabel("Time (sec)", fontsize=14, labelpad=10)
                    num = num + 1
        plt.legend()
        plt.tight_layout()
        plt.show()

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

    def perform_cv(self):
        all_epochs = np.arange(1, self.n_epochs + 1)
        all_train_losses = []
        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = self.train(self.train_loader, epoch)
            all_train_losses.append(epoch_loss)

            # self.test(self.val_loader)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(self.device)
                sample = self.model.decode(sample).cpu()

        all_train_losses = np.array(all_train_losses)
        plt.plot(all_epochs, all_train_losses, label="Train Losses")
        plt.title(self.name)
        plt.xlabel("Epochs")
        plt.ylabel("Average Epoch Losses")
        plt.show()
