import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class VAE_LSTM_ATTN(nn.Module):

    def __init__(self):
        super(VAE_LSTM_ATTN, self).__init__()

        self.input_size = 4  # x.shape[1] * x.shape{2] = 250 * 4
        self.batch_size = 128
        self.batch_first = True

        self.latent_size = 20
        self.hidden_size = 400
        self.num_layers = 1
        self.dropout = 0

        # encoding
        self.lstm_enc = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                dropout=self.dropout, batch_first=True, bidirectional=False)
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_size)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_size)

        # Xavier Initialization should work well with the sigmoid and tanh activation functions within the LSTM units
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

        # decoding
        self.latent_to_hidden = nn.Linear(self.latent_size, self.hidden_size)
        self.input_to_hidden = nn.Linear(self.input_size, self.hidden_size)
        self.lstm_dec = nn.LSTM(input_size=self.hidden_size, hidden_size=self.input_size, num_layers=self.num_layers,
                                dropout=self.dropout, batch_first=True, bidirectional=False)

    def encode(self, x):
        self.seq_len = x.shape[1]

        self.lstm_enc.flatten_parameters()
        lstm_out, (hidden, cell) = self.lstm_enc(x, None)

        # calculate distribution properties for each element of the batch - based on the last lstm cell for each
        if self.batch_first:
            last_state = lstm_out[:, -1, :]
            # hidden = torch.transpose(hidden, 0, 1)
        else:
            last_state = lstm_out[-1, :, :]

        mu = self.hidden_to_mean(last_state)
        logvar = self.hidden_to_logvar(last_state)

        return mu, logvar, lstm_out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # Returns a tensor with the same size as input that is filled with random numbers from a uniform distribution
        # on the interval [0, 1).
        eps = torch.randn_like(std)
        result = mu + eps * std
        return result

    def decode(self, z, target_tensor):
        initial_hidden = self.latent_to_hidden(z)

        # create a new dimension by replicating the hidden state result to correspond to the lstm number of layers
        hidden = torch.cat([initial_hidden] * self.num_layers, 1).view(self.num_layers, -1, self.hidden_size).cuda()
        # first_input = torch.cat()
        cell_state = None

        for di in range(self.seq_len):
            output = torch.unsqueeze(target_tensor[:, di, :], 1)
            output = F.relu(self.input_to_hidden(output))
            # output = F.relu(output)
            output, hidden = self.lstm_dec(output, (hidden, cell_state))
            output = self.softmax(self.out(output[0]))

        # self.lstm_dec.flatten_parameters()
        # lstm_out, (hidden, cell) = self.lstm_dec(hidden_state_exp, None)
        return output

    # def unflatten_dec_input(self, input):
    #     input = torch.cat()

    def forward(self, x):
        mu, logvar, encoder_output = self.encode(x)
        # sample
        z = self.reparameterize(mu, logvar)
        sample = self.decode(z, x)
        return sample, mu, logvar

    # The hidden and cell state dimensions are: (num_layers * num-direction, batch, hidden_size) for each
    def init_hidden(self):
        if self.use_cuda:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float().cuda())
        else:
            return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float(),
                    Variable(torch.zeros(1, self.batch_size, self.hidden_size)).float())
