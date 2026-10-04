import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden=400):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc21 = nn.Linear(hidden, 20)
        self.fc22 = nn.Linear(hidden, 20)
        self.fc3 = nn.Linear(20, hidden)
        self.fc4 = nn.Linear(hidden, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # Reconstruction + KL divergence losses summed over all elements and batch
    BCE = F.binary_cross_entropy(
        recon_x, x.view(-1, recon_x.shape[1]), reduction="sum"
    )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def mnist_transform(scale):
    """ToTensor(), preceded by a bilinear upscale to (28*scale)^2 if scale > 1."""
    if scale == 1:
        return transforms.ToTensor()
    return transforms.Compose([transforms.Resize(28 * scale), transforms.ToTensor()])
