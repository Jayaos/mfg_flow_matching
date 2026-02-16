from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import os
import torch
import numpy as np


def autoencoder_test(model_dir, config_dir, x_input, saving_dir, device):
    """
    test autoencoder by visualizing input and reconstructed output

    Args
    ----
        x_input: (batch_size, channel, h, w)
    """
    num_imgs = x_input.shape[0]

    config = OmegaConf.load(config_dir)
    autoencoder = instantiate_from_config(config.model)

    # Load weights
    ckpt = torch.load(model_dir, map_location=device)
    autoencoder.load_state_dict(ckpt["state_dict"], strict=False)
    autoencoder.eval()
    autoencoder.to(device)

    x_input = x_input.to(device)
    with torch.no_grad():
        reconstructed, posterior = autoencoder(x_input)

    image_list = []

    for i in range(num_imgs):
        image_list.append(x_input[i])

    for i in range(num_imgs):
        image_list.append(reconstructed[i])

    display_mult_images(image_list, rows=2, cols=num_imgs, saving=saving_dir)


def autoencoder_decode(autoencoder, x_input, rescale_factor=None, batch_size=None, device="cpu"):
    """
    x_input: (data_size, dims...)
    """

    if rescale_factor:
        x = x_input * rescale_factor
    else:
        x = x_input

    autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        if batch_size:
            num_batches = int(np.ceil(int(x.shape[0]) / batch_size))
            decoded = []

            for i in range(num_batches):
                x_batch = x[i*batch_size:(i+1)*batch_size]
                x_batch = x_batch.to(device)
                decoded_batch = autoencoder.decode(x_batch)
                decoded.append(decoded_batch)
            decoded = torch.cat(decoded, dim=0)

        else:
            x = x.to(device)
            decoded = autoencoder.decode(x)

    return decoded


def convert_logvar_to_std(logvar):

    logvar = torch.clamp(logvar, -30.0, 20.0)
    return torch.exp(0.5 * logvar)
