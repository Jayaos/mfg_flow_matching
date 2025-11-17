import pickle
import torch
import numpy as np
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


def initialize_linear_interpolant(p_training, timesteps):

    q_target = torch.randn_like(p_training)
    X_bar = torch.zeros(len(timesteps), p_training.size(0), p_training.size(1))

    for i in range(len(timesteps)):
        X_bar[i] = (1 - timesteps[i])*p_training + timesteps[i]*q_target

    return X_bar


def initialize_X_bar_image(p_samples, q_samples, timesteps):

    X_bar = torch.zeros(len(timesteps), 
                        p_samples.size(0), 
                        p_samples.size(1), 
                        p_samples.size(2),
                        p_samples.size(3))

    for i in range(len(timesteps)):
        X_bar[i] = (1 - timesteps[i])*p_samples + timesteps[i]*q_samples

    return X_bar


def batched_odeint(velocity_field, x0, timesteps, batch_size, ode_solver, device):

    results = []
    for i in range(0, len(x0), batch_size):
        x0_batch = x0[i:i+batch_size].to(device)
        out = odeint(velocity_field, x0_batch, timesteps.to(device), method=ode_solver)
        results.append(out.cpu())
        
    return torch.cat(results, dim=1)  # concatenate over training_size dimension


def compute_fid(velocity_field, 
                vae_model, 
                rescale_factor, 
                x_bar,
                q_test,
                q_test_dataloader, 
                timesteps, 
                odeint_batch_size, 
                ode_solver, 
                device):
    # x_bar: trajectories optimized by particle optimization, (data_size, len(timesteps), dims...)
    
    x_init = x_bar[:, 0, :, :, :]
    x_bar_endpoint = x_bar[:, -1, :, :, :]

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
    
    decoded_samples_x_bar_endpoint = autoencoder_decode(vae_model, 
                                         x_bar_endpoint, 
                                         rescale_factor=rescale_factor,
                                         batch_size=odeint_batch_size,
                                         device=device)
    
    decoded_samples_test = autoencoder_decode(vae_model, 
                                         q_test, 
                                         rescale_factor=rescale_factor,
                                         batch_size=odeint_batch_size,
                                         device=device)
    
    # compute inception features from translated images
    rec_mu_velocity_field, rec_sigma_velocity_field = get_inception_features(DataLoader(TensorDataset(decoded_samples_velocity_field), 
                                                                batch_size=odeint_batch_size),
                                                                dims=2048,
                                                                device=device)
    
    rec_mu_x_bar_endpoint, rec_sigma_x_bar_endpoint = get_inception_features(DataLoader(TensorDataset(decoded_samples_x_bar_endpoint), 
                                                                batch_size=odeint_batch_size),
                                                                dims=2048,
                                                                device=device)
    
    rec_mu_test, rec_sigma_test = get_inception_features(DataLoader(TensorDataset(decoded_samples_test), 
                                                                batch_size=odeint_batch_size),
                                                                dims=2048,
                                                                device=device)
    
    # compute inception features from raw test images
    test_true_mu, test_true_sigma = get_inception_features(q_test_dataloader, 
                                                           dims=2048, 
                                                           device=device)

    # conpute FID
    particle_optimization_test_fid = compute_frechet_distance(rec_mu_x_bar_endpoint, 
                                   rec_sigma_x_bar_endpoint, 
                                   test_true_mu, 
                                   test_true_sigma)
    
    velocity_field_test_fid = compute_frechet_distance(rec_mu_velocity_field, 
                                   rec_sigma_velocity_field, 
                                   test_true_mu, 
                                   test_true_sigma)
    
    particle_optimization_rec_test_fid = compute_frechet_distance(rec_mu_x_bar_endpoint, 
                                   rec_sigma_x_bar_endpoint, 
                                   rec_mu_test, 
                                   rec_sigma_test)
    
    velocity_field_rec_test_fid = compute_frechet_distance(rec_mu_velocity_field, 
                                   rec_sigma_velocity_field, 
                                   rec_mu_test, 
                                   rec_sigma_test)
    
    velocity_field_particle_optimization_fid = compute_frechet_distance(rec_mu_x_bar_endpoint, 
                                   rec_sigma_x_bar_endpoint, 
                                   rec_mu_velocity_field, 
                                   rec_sigma_velocity_field)
    
    return particle_optimization_test_fid, velocity_field_test_fid, \
            particle_optimization_rec_test_fid, velocity_field_rec_test_fid, \
            velocity_field_particle_optimization_fid


def compute_test_fid(velocity_field, 
                vae_model, 
                rescale_factor, 
                p_test,
                q_test_dataloader, 
                timesteps, 
                odeint_batch_size, 
                ode_solver, 
                device):

    with torch.no_grad():  # Don't track gradients for ODE solving
        if odeint_batch_size:
            x_trajectory = batched_odeint(velocity_field, 
                                          p_test, 
                                          timesteps,
                                          odeint_batch_size,
                                          ode_solver=ode_solver,
                                          device=device)
        else:
            x_trajectory = odeint(velocity_field, 
                                  p_test.to(device),
                                  timesteps, 
                                  method=ode_solver)
    
    decoded_samples = autoencoder_decode(vae_model, 
                                         x_trajectory[-1, :, :, :, :], 
                                         rescale_factor=rescale_factor,
                                         batch_size=odeint_batch_size,
                                         device=device)
    
    
    # compute inception features from translated images
    rec_mu_velocity_field, rec_sigma_velocity_field = get_inception_features(DataLoader(TensorDataset(decoded_samples), 
                                                                batch_size=odeint_batch_size),
                                                                dims=2048,
                                                                device=device)
    
    # compute inception features from raw test images
    test_true_mu, test_true_sigma = get_inception_features(q_test_dataloader, 
                                                           dims=2048, 
                                                           device=device)

    # conpute FID
    test_fid = compute_frechet_distance(rec_mu_velocity_field, 
                                   rec_sigma_velocity_field, 
                                   test_true_mu, 
                                   test_true_sigma)
    
    
    return test_fid


def compute_fid_endpoint(vae_model, 
                         rescale_factor, 
                         x, 
                         test_dataloader, 
                         batch_size, 
                         device):
    
    decoded_samples = autoencoder_decode(vae_model, 
                                         x, 
                                         rescale_factor=rescale_factor,
                                         batch_size=batch_size,
                                         device=device)
    
    # compute inception features from translated images
    rec_mu, rec_sigma = get_inception_features(DataLoader(TensorDataset(decoded_samples), 
                                                                batch_size=batch_size),
                                                                dims=2048,
                                                                device=device)
    
    # compute inception features from true images
    test_true_mu, test_true_sigma = get_inception_features(test_dataloader, 
                                                           dims=2048, 
                                                           device=device)

    # conpute FID
    test_fid = compute_frechet_distance(rec_mu, 
                                        rec_sigma, 
                                        test_true_mu, 
                                        test_true_sigma)
    
    return test_fid


def compute_l2uvp_cos_forward(benchmark, velocity_field, num_timesteps, odeint_batch_size, ode_solver, size, device):

    p = benchmark.input_sampler.sample(size)
    q = benchmark.map_fwd(p, nograd=True)

    timesteps = torch.linspace(0, 1, num_timesteps)

    with torch.no_grad():  # Don't track gradients for ODE solving
        if odeint_batch_size:
            X_trajectory = batched_odeint(velocity_field, 
                                   p, 
                                    timesteps,
                                    odeint_batch_size,
                                    ode_solver,
                                    device=device)
        else:
            X_trajectory = odeint(velocity_field, p, timesteps, method=ode_solver) 
        # (len(timesteps), training_size, dims, ...)

    p_1 = X_trajectory[-1, :, :]
    p_1 = p_1.to(device)

    with torch.no_grad():
        L2_UVP_fwd = 100 * (((q - p_1) ** 2).sum(dim=1).mean() / benchmark.output_sampler.var).item()
        cos_fwd = (((q - p) * (p_1 - p)).sum(dim=1).mean() / \
                (np.sqrt((2 * benchmark.cost) * ((p_1 - p) ** 2).sum(dim=1).mean().item()))).item()
    
    return L2_UVP_fwd, cos_fwd

@torch.no_grad()
def compute_path_energy(velocity_field, x_init, num_timesteps, odeint_batch_size, ode_solver, device):

    timesteps = torch.linspace(0, 1, num_timesteps).to(device)

    # Don't track gradients for ODE solving
    if odeint_batch_size:
        X_bar = batched_odeint(velocity_field, 
                                x_init, 
                                timesteps,
                                odeint_batch_size,
                                ode_solver,
                                device=device)
    else:
        X_bar = odeint(velocity_field, x_init, timesteps, method=ode_solver) 

    T, N, D = X_bar.shape
    X_bar = X_bar.to(device)

    # v(t_k, Z_{t_k})
    v_vals = []
    for k in range(T):
        tk = torch.full((N, 1), timesteps[k].item(), device=device).flatten()
        v_vals.append(velocity_field(tk, X_bar[k]))   # (N, D)
    v_vals = torch.stack(v_vals, dim=0)                # (T, N, D)

    v2 = (v_vals**2).sum(dim=2)                        # (T, N)
    dt = timesteps[1:] - timesteps[:-1]    # (T-1,)
    estimated = 0.5 * (v2[:-1] + v2[1:]) * dt.view(-1, 1)  # estimating intergal

    return estimated.sum(dim=0).mean()                      # E over batch