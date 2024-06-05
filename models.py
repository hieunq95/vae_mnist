import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, latent_dim=20):
        super().__init__()
        self.input_dim = input_dim  #
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Define fully-connected (fc) layers. Pay attention to the change of input-output sizes
        # Model architecture is visualized in the README.md file
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)  # output layer has the same size as the input

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, std):
        if self.training:
            std = torch.exp(0.5*std)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        # Take z as input and output a probability (via sigmoid() activation) on each dimension.
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # Note that we take input data x and output a probability (not a pixel value)
        z_mu, z_std = self.encode(x.view(-1, self.input_dim))  # flatten input data
        z = self.reparameterize(z_mu, z_std)
        x_probs = self.decode(z)
        return x_probs, z_mu, z_std


class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, latent_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return h2

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dim))
        x_recon = self.decode(z)
        return x_recon


