import torch
from abc import ABC, abstractmethod
from torch import nn, Tensor


class VelocityField(ABC, nn.Module):
    """Abstract base class for velocity fields."""

    @abstractmethod
    def forward(self, t, x):
        ...


class ConcatLinear(nn.Module):
    def __init__(self, input_dim: int, time_dim: int, output_dim: int, activation=None):
        super(ConcatLinear, self).__init__()
        self.linear = nn.Linear(input_dim + time_dim, output_dim)
        self.activation = activation

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        xt = torch.cat([x, t], dim=-1) 
        out = self.linear(xt)

        if self.activation is not None:
            out = self.activation(out)

        return out
        

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


class MLPVelocityField(VelocityField):
    def __init__(self, input_dim: int, time_dim: int, hidden_dims: list, layer_type: str, activation=None):
        """
        Generalized MLP with ConcatLinear or ConcatSquashLinear layers 
        to model vector field of continuous normalizing flow models

        Args
        ----
        :param input_dim: Input feature dimension
        :param time_dim: Time input dimension
        :param hidden_dims: List of hidden layer dimensions
        :param layer_type: Type of layers to use ('linear' or 'squash')
        :param activation: Activation function (default: None)

        Returns
        -------
        """
        super(MLPVelocityField, self).__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims
        self.layer_type = layer_type
        self.activation = activation

        # Validate layer type
        layer_cls = {"concatlinear": ConcatLinear,}.get(layer_type)
        if layer_cls is None:
            raise ValueError(f"Invalid layer type: {layer_type}. Choose 'linear' or 'squash'.")

        layers = nn.ModuleList()

        # First layer
        self.first_layer = layer_cls(input_dim, time_dim, hidden_dims[0], activation)

        # Hidden layers
        if len(hidden_dims) > 1:
            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers.append(Linear(in_dim, out_dim, activation))
        else:
            pass

        # Last layer
        layers.append(Linear(hidden_dims[-1], input_dim, activation=None))

        self.layers = nn.Sequential(*layers)

    def forward(self, t, x):
        """
        Args
        ----
            x: input tensor, (batch_size, timesteps, input_dim) or (batch_size, input_dim)
            t: time tensor, (timesteps)

        Returns
        -------
            output tensor, (batch_size, input_dim)
        """
        if x.dim() == 3:
            # when we need velocity field for particle at each time
            t = t.unsqueeze(1).transpose(1,0).expand(x.size(0), x.size(1)).unsqueeze(2) # (batch_size, len(timesteps), 1)
            h = self.first_layer(x, t)
        elif x.dim() == 2:
            # for ODE integration from the beginning point
            t = t.repeat(x.size(0)).unsqueeze(-1)
            h = self.first_layer(x, t)

        return self.layers(h)
    

class ConvCat2d_base(nn.Module):
    def __init__(self, in_dim, out_dim, ksize, stride, transpose = False, use_t = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transpose = transpose
        in_dim = in_dim + 1 if use_t else in_dim
        pad_size = 1
        if transpose:
            self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=pad_size)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=pad_size)

    def forward(self, t, x):
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(x.shape[0])
            tt = t.view(-1, 1, 1, 1).repeat(1, 1, *x.shape[2:])
        elif t.dim() == 1:
            tt = t.view(-1, 1, 1, 1).repeat(1, 1, *x.shape[2:])
        else:
            # Assume always (bsize, a, b, c)
            tt = t
        ttx = torch.cat([tt, x], 1)
        return self.conv(ttx)
    

class ConvCat2d(nn.Module):
    # Simple time-parameterized net, with 2d conv and conv transpose
    def __init__(self, input_dim, enc_dims, dec_dims, ksizes, strides):
        super().__init__()
        self.enc_dims = [input_dim,] + enc_dims
        self.dec_dims = dec_dims + [input_dim,]
        self.activation = nn.ReLU()
        self.ksizes = ksizes
        self.strides = strides
        self.build_layers()

    def build_layers(self):
        self.layers = nn.ModuleList()
        i_ = 0
        for i in range(len(self.enc_dims)-1):
            in_dim, out_dim = self.enc_dims[i], self.enc_dims[i+1]
            ksize, stride = self.ksizes[i_], self.strides[i_]
            self.layers.append(ConvCat2d_base(in_dim, out_dim, ksize, stride))
            i_ += 1
        for i in range(len(self.dec_dims)-1):
            in_dim, out_dim = self.dec_dims[i], self.dec_dims[i+1]
            ksize, stride = self.ksizes[i_], self.strides[i_]
            self.layers.append(ConvCat2d_base(in_dim, out_dim, ksize, stride, transpose=True))
            i_ += 1
            
    def forward(self, t, x):
        for i, layer in enumerate(self.layers):
            x = layer(t, x)
            if i < len(self.layers)-1:
                x = self.activation(x)
        return x


class ConvVelocityField(VelocityField):

    def __init__(self, input_dim, enc_dims, dec_dims, ksizes, strides):

        super(ConvVelocityField, self).__init__()

        self.convcat2d = ConvCat2d(input_dim, enc_dims, dec_dims, ksizes, strides)


    def forward(self, t, x):

        return self.convcat2d(t, x)