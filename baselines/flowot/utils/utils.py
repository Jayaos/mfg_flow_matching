import pickle
import torch
import os
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from evaluation.fid.eval import get_inception_features, compute_frechet_distance
from ldm.models.utils import autoencoder_decode
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)


def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)


def batched_odeint(velocity_field, x0, timesteps, batch_size, ode_solver, device):

    results = []
    for i in range(0, len(x0), batch_size):
        x0_batch = x0[i:i+batch_size].to(device)
        out = odeint(velocity_field, x0_batch, timesteps.to(device), method=ode_solver)
        results.append(out.cpu())
        
    return torch.cat(results, dim=1)  # concatenate over training_size dimension


def plot_losses(kl_loss, wasserstein_loss, refinement_loss, saving_dir, filename):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Example: plotting three different loss records
    axes[0].plot(kl_loss)
    axes[0].set_title("KL loss")

    axes[1].plot(wasserstein_loss)
    axes[1].set_title("Wasserstein loss")

    axes[2].plot(refinement_loss)
    axes[2].set_title("Refinement loss")

    plt.tight_layout()
    plt.savefig(os.path.join(saving_dir, filename), dpi=300)


def compute_fid(velocity_field, 
                vae_model, 
                rescale_factor, 
                x_init,
                test_dataloader, 
                timesteps, 
                odeint_batch_size, 
                ode_solver, 
                device):

    with torch.no_grad():  # Don't track gradients for ODE solving
        if odeint_batch_size:
            x_trajectory = batched_odeint(velocity_field, 
                                          x_init, 
                                          timesteps,
                                          odeint_batch_size,
                                          ode_solver=ode_solver,
                                          device=device)
        else:
            x_trajectory = odeint(velocity_field, 
                                  x_init.to(device),
                                  timesteps, 
                                  method=ode_solver)
            
    decoded_samples_velocity_field = autoencoder_decode(vae_model, 
                                         x_trajectory[-1, :, :, :, :], 
                                         rescale_factor=rescale_factor,
                                         batch_size=odeint_batch_size,
                                         device=device)

    
    # compute inception features from translated images
    rec_mu_velocity_field, rec_sigma_velocity_field = get_inception_features(DataLoader(TensorDataset(decoded_samples_velocity_field), 
                                                                batch_size=odeint_batch_size),
                                                                dims=2048,
                                                                device=device)
    
    # compute inception features from true images
    test_true_mu, test_true_sigma = get_inception_features(test_dataloader, 
                                                           dims=2048, 
                                                           device=device)

    # conpute FID
    velocity_field_rec_test_fid = compute_frechet_distance(rec_mu_velocity_field, 
                                   rec_sigma_velocity_field, 
                                   test_true_mu, 
                                   test_true_sigma)
    
    return velocity_field_rec_test_fid
