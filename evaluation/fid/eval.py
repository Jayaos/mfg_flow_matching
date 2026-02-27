from .model import InceptionV3
import torch
import numpy as np
import scipy.linalg as linalg
from tqdm import tqdm
import gc


def _freeze_model_parameters(model):
    for p in model.parameters():
        p.requires_grad_(False)


def get_inception_features(data_loader, dims=2048, device="cpu"):
    """
    obtain mean and covariance matrix from inception features given data_loader
    """

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inceptionv3_model = InceptionV3([block_idx]).to(device)
    _freeze_model_parameters(inceptionv3_model)
    
    inception_features = []

    inceptionv3_model.eval()
    with torch.no_grad():
        for x in tqdm(data_loader):

            if isinstance(x, torch.Tensor):
                # if data_loader uses images
                x = (x+1)/2
                x = x.to(device)
                features = inceptionv3_model(x)[0].view(x.shape[0], -1) # (batch_size, dims)
                inception_features.append(features.cpu())
            elif isinstance(x, list):
                # if data_loader uses reconstruced data
                x = x[0]
                x = (x+1)/2
                x = x.to(device)
                features = inceptionv3_model(x)[0].view(x.shape[0], -1) # (batch_size, dims)
                inception_features.append(features.cpu())

    if device == "cuda":
        print(f"[GPU mem] allocated={torch.cuda.memory_allocated()/1024**2:.1f} MB | "
                f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MB", flush=True)

    inception_features = np.vstack(inception_features)
    mu, sigma = np.mean(inception_features, axis=0), np.cov(inception_features, rowvar=False)
    gc.collect() 
    torch.cuda.empty_cache()

    return mu, sigma


def build_reconstructed_data(vae_model, data_loader, device):

    vae_model.to(device)
    reconstructed_x = []

    vae_model.eval()
    with torch.no_grad():
        for x in tqdm(data_loader):

            x = x.to(device)
            reconstructed_x.append(vae_model(x)[0].cpu())

    reconstructed_x = torch.vstack(reconstructed_x)

    return reconstructed_x


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

