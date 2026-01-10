import numpy as np
import torch
import math
from PIL import Image
from torch.utils.data import DataLoader


def load_shoebags_latent_params(dataset_config):

    shoes_latent_params = torch.load(dataset_config.shoes_latent_params_dir)
    bags_latent_params = torch.load(dataset_config.bags_latent_params_dir)
    shoes_mu, shoes_logvar = torch.chunk(shoes_latent_params, 2, dim=1) # (data_size, channel/2, w, h)
    bags_mu, bags_logvar = torch.chunk(bags_latent_params, 2, dim=1) # (data_size, channel/2, w, h)

    return shoes_mu, shoes_logvar, bags_mu, bags_logvar


def compute_rescale_factor(mu, std):

    total_sample = min(len(mu), 100000) # Use at most 100k samples to estimate the std
    z = (sample_latent_image(mu[:total_sample], std[:total_sample]))

    return z.flatten().std()


def sample_latent_image(mu, std, rescale=None):

    z = torch.randn_like(mu)

    if rescale:
        return (mu + std*z)/rescale  
    else:
        return mu + std*z
    

def generate_latent_image_data(p_mu, p_std, q_mu, q_std, train_ratio, seed, p_rescale_factor, q_rescale_factor):

    torch.manual_seed(seed)  # for reproducibility

    if p_mu.size(0) > q_mu.size(0):
        p_multiplier = 1
        q_multiplier = (p_mu.size(0) // q_mu.size(0))+1
    else:
        p_multiplier = (q_mu.size(0) // p_mu.size(0))+1
        q_multiplier = 1        

    p_size = p_mu.size(0)
    p_indices = torch.randperm(p_size)
    p_train_size = int(train_ratio * p_size)
    p_train_idx = p_indices[:p_train_size]
    p_test_idx = p_indices[p_train_size:]

    generated_p_train_samples = []
    for _ in range(p_multiplier):
        generated_p_train_samples.append(sample_latent_image(p_mu[p_train_idx], 
                                                              p_std[p_train_idx], 
                                                              rescale=p_rescale_factor))
    
    generated_p_test_samples = []
    for _ in range(p_multiplier):
        generated_p_test_samples.append(sample_latent_image(p_mu[p_test_idx], 
                                                              p_std[p_test_idx], 
                                                              rescale=p_rescale_factor))
    
    q_size = q_mu.size(0)
    q_indices = torch.randperm(q_size)
    q_train_size = int(train_ratio * q_size)
    q_train_idx = q_indices[:q_train_size]
    q_test_idx = q_indices[q_train_size:]

    generated_q_train_samples = []
    for _ in range(q_multiplier):
        generated_q_train_samples.append(sample_latent_image(q_mu[q_train_idx], 
                                                              q_std[q_train_idx], 
                                                              rescale=q_rescale_factor))
    
    generated_q_test_samples = []
    for _ in range(q_multiplier):
        generated_q_test_samples.append(sample_latent_image(q_mu[q_test_idx], 
                                                              q_std[q_test_idx], 
                                                              rescale=q_rescale_factor))
        
    generated_p_train_samples = torch.cat(generated_p_train_samples, dim=0)
    generated_p_test_samples = torch.cat(generated_p_test_samples, dim=0)
    generated_q_train_samples = torch.cat(generated_q_train_samples, dim=0)
    generated_q_test_samples = torch.cat(generated_q_test_samples, dim=0)

    train_size = min(generated_p_train_samples.size(0), generated_q_train_samples.size(0))
    test_size = min(generated_p_test_samples.size(0), generated_q_test_samples.size(0))

    # both p and q samples have the same size 
    generated_p_train_samples = generated_p_train_samples[:train_size]
    generated_q_train_samples = generated_q_train_samples[:train_size]
    generated_p_test_samples = generated_p_test_samples[:test_size]
    generated_q_test_samples = generated_q_test_samples[:test_size]
        
    return generated_p_train_samples, generated_p_test_samples, generated_q_train_samples, generated_q_test_samples


def random_sample_replicated_latent_image_data(p_mu, p_std, q_mu, q_std, train_ratio, seed, 
                                               p_rescale_factor, q_rescale_factor, sample_num, replication):
    """
    randomly sample given number of latent image with given number of replication
    """
    torch.manual_seed(seed)  # for reproducibility

    if p_mu.size(0) > q_mu.size(0):
        p_multiplier = 1
        q_multiplier = (p_mu.size(0) // q_mu.size(0))+1
    else:
        p_multiplier = (q_mu.size(0) // p_mu.size(0))+1
        q_multiplier = 1        

    p_size = p_mu.size(0)
    p_indices = torch.randperm(p_size)
    p_train_size = int(train_ratio * p_size)
    p_train_idx = p_indices[:p_train_size]
    p_test_idx = p_indices[p_train_size:]

    p_test_mu_sample = p_mu[p_test_idx][:sample_num]
    p_test_std_sample = p_std[p_test_idx][:sample_num]

    generated_p_test_samples = []
    for _ in range(replication):
        for i in range(sample_num):
            generated_p_test_samples.append(sample_latent_image(p_test_mu_sample[i], 
                                                                p_test_std_sample[i], 
                                                                rescale=p_rescale_factor).unsqueeze(0))
            
    # restore true latent image by deterministic mapping
    p_test_true_sample = (p_test_mu_sample + p_test_std_sample)/p_rescale_factor
    # generated_p_test_samples: (replication*sample_num, latent channel, latent w, latent h)
    generated_p_test_samples = torch.cat(generated_p_test_samples, dim=0)
    
    return p_test_true_sample, generated_p_test_samples


class DataLoaderIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iterator = iter(self.dataloader)
        self.epoch = 0
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._iterator = iter(self.dataloader)  # restart for next epoch
            batch = next(self._iterator)

        self.step += 1

        return batch
    
    
class TrajectoryDataLoader:
    def __init__(self, xbar_p, xbar_trajectory, batch_size, shuffle=True):
        self.xbar_p = xbar_p # this is not trainable
        self.xbar_trajectory = xbar_trajectory # this is trainable
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_particles = xbar_trajectory.size(1)
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_particles)
        else:
            indices = torch.arange(self.num_particles)
        
        # Yield batches that cover ALL particles
        for i in range(0, self.num_particles, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.xbar_p[:, batch_indices, :], self.xbar_trajectory[:, batch_indices, :]
        
    def __len__(self):
        return (self.num_particles + self.batch_size - 1) // self.batch_size
    

class ImageTrajectoryDataLoader:
    def __init__(self, xbar_p, xbar_trajectory, batch_size, shuffle=True):
        self.xbar_p = xbar_p # this is not trainable
        self.xbar_trajectory = xbar_trajectory # this is trainable
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_particles = xbar_trajectory.size(1)
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_particles)
        else:
            indices = torch.arange(self.num_particles)
        
        # Yield batches that cover ALL particles
        for i in range(0, self.num_particles, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.xbar_p[:, batch_indices, :], self.xbar_trajectory[:, batch_indices, :]
        
    def __len__(self):
        return (self.num_particles + self.batch_size - 1) // self.batch_size