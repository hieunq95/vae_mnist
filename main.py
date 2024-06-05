import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from torch.nn import functional as F
from models import VAE, Autoencoder
from torchvision.utils import save_image
from matplotlib import pyplot as plt

# torch.manual_seed(123)


def vae_loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL-divergence loss
    return BCE + KLD


def autoencoder_loss_function(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, reduction='sum')


def vae_interpolation(model, x1, x2, steps=32):
    # interpolate between data points x1 and x2 in a number of steps
    z1_mu, z1_std = model.encode(x1.view(-1, model.input_dim))
    z1 = model.reparameterize(z1_mu, z1_std)
    z2_mu, z2_std = model.encode(x2.view(-1, model.input_dim))
    z2 = model.reparameterize(z2_mu, z2_std)

    alphas = np.linspace(0, 1, steps)
    x_recon = torch.zeros(steps, 1, 28, 28)
    for j in range(len(alphas)):
        z_j = alphas[j] * z1 + (1 - alphas[j]) * z2
        x_j = model.decode(z_j)
        x_j = torch.reshape(x_j, (1, 28, 28))
        x_recon[j] = x_j
    save_image(x_recon.view(steps, 1, 28, 28), 'images/vae_interpolation_{}.png'.format(steps))


def autoencoder_interpolation(model, x1, x2, steps=32):
    z1 = model.encode(x1.view(-1, model.input_dim))
    z2 = model.encode(x2.view(-1, model.input_dim))
    alphas = np.linspace(0, 1, steps)
    x_recon = torch.zeros(steps, 1, 28, 28)
    for j in range(len(alphas)):
        z_j = alphas[j] * z1 + (1 - alphas[j]) * z2
        x_j = model.decode(z_j)
        x_j = torch.reshape(x_j, (1, 28, 28))
        x_recon[j] = x_j
    save_image(x_recon.view(steps, 1, 28, 28), 'images/autoencoder_interpolation_{}.png'.format(steps))


def train_vae(model, num_epochs, data_loader, optimizer, log_interval=2):
    print('Start training VAE model:\n{}'.format(model))
    model.train()
    for epoch in range(num_epochs):
        ep_loss = []
        for batch_id, (data, _) in enumerate(data_loader):
            x_batch = torch.squeeze(data).view(-1, model.input_dim)  # flatten
            optimizer.zero_grad()
            x_probs, mu, log_var = model(x_batch)
            loss = vae_loss_function(x_probs, x_batch, mu, log_var)
            loss.backward()
            ep_loss.append(loss.item())
            optimizer.step()
        if epoch % log_interval == 0:
            print('Epoch: {}, Ep_Loss: {}'.format(epoch, np.mean(ep_loss)))
    torch.save(model.state_dict(), 'model_parameters/vae_param')
    print('Saved model parameters to model_parameters/vae_param')


def test_vae(model, num_epochs, data_loader):
    # Test reconstruction accuracy and generate new data samples
    print('\nTesting VAE model:\n{}'.format(model))
    model.load_state_dict(torch.load('model_parameters/vae_param'))
    model.eval()
    for epoch in range(num_epochs):
        ep_loss = []
        for data, _ in data_loader:
            x_batch = torch.squeeze(data).view(-1, model.input_dim)
            recon_x, mu, log_var = model(x_batch)
            loss = vae_loss_function(recon_x, x_batch, mu, log_var)
            ep_loss.append(loss.item())
        print('Epoch: {}, test_loss: {}'.format(epoch, np.mean(ep_loss)))

    print('\nVisualize VAE latent space ...')
    for data, _ in data_loader:
        x_batch = torch.squeeze(data).view(-1, model.input_dim)
        z_mu, z_std = model.encode(x_batch)
        z = model.reparameterize(z_mu, z_std)
        z_0 = []
        z_1 = []
        for z_j in z:
            z_0.append(z_j[0].detach().numpy())
            z_1.append(z_j[1].detach().numpy())
        plt.scatter(z_0, z_1, alpha=0.5)
    plt.xlabel('$z_0$')
    plt.ylabel('$z_1$')
    plt.savefig('images/vae_latent_space.png')
    plt.grid(True)
    plt.close()

    print('\nSampling random data samples:')
    # note that we need to sample from a Gaussian distribution
    num_samples = 16
    z = torch.randn(num_samples, model.latent_dim)
    x_sample = model.decode(z)
    # save images
    save_image(x_sample.view(num_samples, 1, 28, 28), 'images/vae_samples_{}.png'.format(num_samples))

    print('\nData interpolation between two data points:')
    # Get two data points from the test set
    x1_batch, _ = data_loader.dataset[0]
    x2_batch, _ = data_loader.dataset[-1]
    x1 = x1_batch[0]  # first image
    x2 = x2_batch[0]  # second image
    # we project two data points into the latent space of the VAE
    vae_interpolation(model, x1, x2, 24)


def train_autoencoder(model, num_epochs, data_loader, optimizer, log_interval=2):
    print('Start training Autoencoder model:\n{}'.format(model))
    model.train()
    for epoch in range(num_epochs):
        ep_loss = []
        for batch_id, (data, _) in enumerate(data_loader):
            x_batch = torch.squeeze(data).view(-1, model.input_dim)  # flatten
            optimizer.zero_grad()
            x_recon = model(x_batch)
            loss = autoencoder_loss_function(x_recon, x_batch)
            loss.backward()
            ep_loss.append(loss.item())
            optimizer.step()
        if epoch % log_interval == 0:
            print('Epoch: {}, Ep_Loss: {}'.format(epoch, np.mean(ep_loss)))
    torch.save(model.state_dict(), 'model_parameters/autoencoder_param')
    print('Saved model parameters to model_parameters/autoencoder_param')


def test_autoencoder(model, num_epochs, data_loader):
    # Test reconstruction accuracy and generate new data samples
    print('\nTesting Autoencoder model:\n{}'.format(model))
    model.load_state_dict(torch.load('model_parameters/autoencoder_param'))
    model.eval()
    for epoch in range(num_epochs):
        ep_loss = []
        for data, _ in data_loader:
            x_batch = torch.squeeze(data).view(-1, model.input_dim)
            recon_x = model(x_batch)
            loss = autoencoder_loss_function(recon_x, x_batch)
            ep_loss.append(loss.item())
        print('Epoch: {}, test_loss: {}'.format(epoch, np.mean(ep_loss)))

    print('\nVisualize Autoencoder latent space ...')
    for data, _ in data_loader:
        x_batch = torch.squeeze(data).view(-1, model.input_dim)
        z = model.encode(x_batch)
        z_0 = []
        z_1 = []
        for z_j in z:
            z_0.append(z_j[0].detach().numpy())
            z_1.append(z_j[1].detach().numpy())
        plt.scatter(z_0, z_1, alpha=0.5)
    plt.xlabel('$z_0$')
    plt.ylabel('$z_1$')
    plt.savefig('images/autoencoder_latent_space.png')
    plt.grid(True)
    plt.close()

    print('\nSampling random data samples:')
    # note that we need to sample from a Gaussian distribution
    num_samples = 16
    # We use larger z as the latent space of the AE is not Gaussian distributed. The scale factor can be removed
    z = 1 * torch.randn(num_samples, model.latent_dim)
    x_sample = model.decode(z)
    # save images
    save_image(x_sample.view(num_samples, 1, 28, 28), 'images/autoencoder_samples_{}.png'.format(num_samples))

    print('\nData interpolation between two data points:')
    # Get two data points from the test set
    x1_batch, _ = data_loader.dataset[0]
    x2_batch, _ = data_loader.dataset[-1]
    x1 = x1_batch[0]  # first image
    x2 = x2_batch[0]  # second image
    # we project two data points into the latent space of the VAE
    autoencoder_interpolation(model, x1, x2, 24)


if __name__ == '__main__':
    num_epochs = 20
    batch_size = 64
    vae_model = VAE()
    autoencoder = Autoencoder()
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    train_set = datasets.MNIST('mnist_dataset', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('mnist_dataset', train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    train_vae(vae_model, num_epochs, train_loader, vae_optimizer)
    test_vae(vae_model, 5, test_loader)

    train_autoencoder(autoencoder, num_epochs, train_loader, autoencoder_optimizer)
    test_autoencoder(autoencoder, 5, test_loader)





