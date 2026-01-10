import torch
from abc import ABC, abstractmethod
from torch import nn, Tensor


class Classifier(ABC, nn.Module):
    """Abstract base class for classifiers."""

    @abstractmethod
    def forward(self, x):
        ...


class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:

        if self.activation is None:
            return self.linear(x)
        else:
            return self.activation(self.linear(x))


class MLPClassifier(Classifier):
    def __init__(self, input_dim: int, hidden_dims: list, activation=None):
        """
        Args
        ----
        :param input_dim: Input feature dimension
        :param hidden_dims: List of hidden layer dimensions
        :param activation: Activation function (default: None)

        Returns
        -------
        """
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        layers = nn.ModuleList()

        # First layer
        self.first_layer = Linear(input_dim, hidden_dims[0], activation)

        # Hidden layers
        if len(hidden_dims) > 1:
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers.append(Linear(in_dim, out_dim, activation))
        else:
            pass

        # Last layer
        layers.append(Linear(hidden_dims[-1], 1, activation=None))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args
        ----
            x: input tensor, (batch_size, input_dim)

        Returns
        -------
            output tensor, (batch_size, 1)
        """
        h = self.first_layer(x)

        return self.layers(h)


class UNetClassifier(Classifier):
    """
    U-Net based Classifier for image tasks.
    """
    def __init__(self, input_dim, channels, use_bias):
        super(UNetClassifier, self).__init__()
        self.channels = channels
        self.use_bias = use_bias
        padding = 1

        # Encoding layers where the resolution decreases
        ## now channel numbers is fixed to 5
        self.conv1 = nn.Conv2d(input_dim, self.channels[0], 3, stride=1, bias=use_bias, padding=padding)
        self.gnorm1 = nn.GroupNorm(4, num_channels=self.channels[0])
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1], 3, stride=1, bias=use_bias, padding=padding)
        self.gnorm2 = nn.GroupNorm(32, num_channels=self.channels[1])
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2], 3, stride=2, bias=use_bias, padding=padding)
        self.gnorm3 = nn.GroupNorm(32, num_channels=self.channels[2])
        self.conv4 = nn.Conv2d(self.channels[2], self.channels[3], 3, stride=1, bias=use_bias, padding=padding)
        self.gnorm4 = nn.GroupNorm(32, num_channels=self.channels[3])
        self.conv5 = nn.Conv2d(self.channels[3], self.channels[4], 3, stride=1, bias=use_bias, padding=padding)
        self.gnorm5 = nn.GroupNorm(32, num_channels=self.channels[4])
        # Decoding layers where the resolution increases
        self.tconv5 = nn.ConvTranspose2d(self.channels[4], self.channels[3], 3, stride=1,
                                         bias=use_bias, padding=padding)
        self.tgnorm5 = nn.GroupNorm(32, num_channels=self.channels[3])
        self.tconv4 = nn.ConvTranspose2d(self.channels[3], self.channels[2], 3, stride=1,
                                            bias=use_bias, padding=padding)
        self.tgnorm4 = nn.GroupNorm(32, num_channels=self.channels[2])
        self.tconv3 = nn.ConvTranspose2d(self.channels[2] + self.channels[2], self.channels[1], 4,
                                            stride=2, bias=use_bias, padding=padding)
        self.tgnorm3 = nn.GroupNorm(32, num_channels=self.channels[1])
        self.tconv2 = nn.ConvTranspose2d(self.channels[1] + self.channels[1], self.channels[0], 3,
                                            stride=1, bias=use_bias, padding=padding)
        self.tgnorm2 = nn.GroupNorm(32, num_channels=self.channels[0])
        self.tconv1 = nn.ConvTranspose2d(self.channels[0] + self.channels[0], input_dim, 3, stride=1, padding=padding)

        # The final fc layer
        img_size = input_dim*8**2
        self.out_fc = nn.Linear(img_size, 1)
        hid_size = img_size
        fc_layers = [nn.Linear(img_size, hid_size), nn.ReLU(), 
                     nn.Linear(hid_size, hid_size), nn.ReLU(), 
                     nn.Linear(hid_size, 1)]
        self.out_fc = nn.Sequential(*fc_layers)
            
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        n = x.size(0)
        # Encoding path
        h1 = self.conv1(x)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        h5 = self.conv5(h4)
        h5 = self.gnorm5(h5)
        h5 = self.act(h5)
        h4 = self.tconv5(h5)
        h4 = self.tgnorm5(h4)
        h4 = self.act(h4)
        # Decoding path
        h = self.tconv4(h4)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = h.view(n, -1)
        out = self.out_fc(h)
        return out
