import torch.nn as nn


class Discriminator(nn.Module):
    """
    Generative Adverserial Network Discriminator Class

    Takes in an image as input and outputs a probability indicating whether or not
    the input belongs to a the real data distribution.
    """

    def __init__(self, img_size, hidden_size):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(img_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of a discriminator

        :param x: Image tensor
        :return: Float in range [0, 1] - probability score
        """
        # Resize x from a H x W img to a vector
        x = x.view(-1, self.img_size)
        return self.model(x).clamp(1e-9)


class Generator(nn.Module):
    """
    Generative Adversarial Network Generator Class

    Takes in a latent vector z and returns a vector in
    the same image space that the discriminator is trained on.

    """

    def __init__(self, img_size, latent_size, hidden_size):
        super(Generator, self).__init__()

        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass of a generator

        :param z: Latent space vector - size: batch_size x latent_size
        :return: Tensor of self.img_size
        """
        return self.model(z)