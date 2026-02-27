import numpy as np
import torch
import math
from PIL import Image
from torch.utils.data import DataLoader
from config import ShoebagsDatasetConfig, CelebADatasetConfig
from ldm.data.shoebags import Shoes, Bags
from ldm.data.celeba import CelebATestMale, CelebATestFemale, CelebAValidMale, CelebAValidFemale
from ldm.data.base import ImageDefaultDataset
from ldm.models.utils import convert_logvar_to_std


def generate_checkerboard_2d(data_size, img_dir, seed=None):

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def gen_data_from_img(image_mask, train_data_size):
        def sample_data(train_data_size):
            inds = np.random.choice(int(probs.shape[0]), int(train_data_size), p=probs)
            m = means[inds]
            samples = np.random.randn(*m.shape) * std + m
            return samples

        img = image_mask
        h, w = img.shape
        xx = np.linspace(-4, 4, w)
        yy = np.linspace(-4, 4, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        means = np.concatenate([xx, yy], 1)  # (h*w, 2)
        img = img.max() - img
        probs = img.reshape(-1) / img.sum()
        std = np.array([8 / w / 2, 8 / h / 2])
        full_data = sample_data(train_data_size)
        
        return torch.from_numpy(full_data).to(dtype=torch.float32)

    image_mask = np.array(Image.open(img_dir).rotate(180).transpose(Image.FLIP_LEFT_RIGHT).convert('L'))
    dataset = gen_data_from_img(image_mask, data_size)
    
    # Add a small bit of noise to the dataset
    h = 0.0025
    Z = torch.randn_like(dataset)
    scale_X = math.exp(-h)
    scale_Z = math.sqrt(1-scale_X**2)
    
    return dataset * scale_X + Z * scale_Z


def generate_toy_data(dataset_config, seed=None):

    if dataset_config.data_name in ["checkerboard-2d", "checkerboard_2d"]:
        data = generate_checkerboard_2d(dataset_config.training_size + dataset_config.test_size, 
                                        dataset_config.img_dir, 
                                        seed)
        # split like this due to seed
        training_data = data[:dataset_config.training_size]
        test_data = data[dataset_config.training_size:]

    elif dataset_config.data_name in ["Gaussian", "normal"]:
        if seed:
            torch.manual_seed(seed)
        training_data = torch.randn(dataset_config.training_size, 2)
        test_data = torch.randn(dataset_config.test_size, 2)

    return training_data, test_data


def load_image_dataset(config, dataset_config):
    """
    load image dataset based on the dataset_config argument
    training and test set
    """

    if isinstance(dataset_config, ShoebagsDatasetConfig):
        image_dim = (64, 64, 3)
        training_bags_mu, training_bags_logvar, training_shoes_mu, training_shoes_logvar, \
            test_bags_mu, test_bags_logvar, test_shoes_mu, test_shoes_logvar= load_shoebags_latent_params(dataset_config)
        
        training_bags_std = convert_logvar_to_std(training_bags_logvar)
        training_shoes_std = convert_logvar_to_std(training_shoes_logvar)
        test_bags_std = convert_logvar_to_std(test_bags_logvar)
        test_shoes_std = convert_logvar_to_std(test_shoes_logvar)

        # P: bags Q: shoes
        p_rescale_factor = compute_rescale_factor(torch.cat([training_bags_mu, test_bags_mu], dim=0), 
                                                     torch.cat([training_bags_std, test_bags_std], dim=0))
        q_rescale_factor = compute_rescale_factor(torch.cat([training_shoes_mu, test_shoes_mu], dim=0), 
                                                     torch.cat([training_shoes_std, test_shoes_std], dim=0))
        
        p_training, q_training = generate_latent_image_data(training_bags_mu, training_bags_std,
                                                            training_shoes_mu, training_shoes_std,
                                                            p_rescale_factor, q_rescale_factor,
                                                            augmentation=True)
        p_test, q_test = generate_latent_image_data(test_bags_mu, test_bags_std,
                                                    test_shoes_mu, test_shoes_std,
                                                    p_rescale_factor, q_rescale_factor,
                                                    augmentation=False)
        
        # load shoebags test dataset
        p_test_dataset = Bags(dataset_config.bags_data_dir, 
                              dataset_config.train_ratio, 
                              dataset_config.seed, 
                              complement=True)
        q_test_dataset = Shoes(dataset_config.shoes_data_dir, 
                               dataset_config.train_ratio, 
                               dataset_config.seed, 
                               complement=True)
        p_test_dataloader = DataLoader(ImageDefaultDataset(p_test_dataset), 
                                        batch_size=config.odeint_minibatch, 
                                        shuffle=False)
        q_test_dataloader = DataLoader(ImageDefaultDataset(q_test_dataset), 
                                        batch_size=config.odeint_minibatch, 
                                        shuffle=False)
        
    elif isinstance(dataset_config, CelebADatasetConfig):
        image_dim = (64, 64, 3)
        training_male_mu, training_male_logvar, training_female_mu, training_female_logvar, \
            test_male_mu, test_male_logvar, test_female_mu, test_female_logvar = load_celeba_latent_params(dataset_config)
        
        training_male_std = convert_logvar_to_std(training_male_logvar)
        training_female_std = convert_logvar_to_std(training_female_logvar)
        test_male_std = convert_logvar_to_std(test_male_logvar)
        test_female_std = convert_logvar_to_std(test_female_logvar)

        # P: male, Q: female
        p_rescale_factor = compute_rescale_factor(torch.cat([training_male_mu, test_male_mu], dim=0), 
                                                     torch.cat([training_male_std, test_male_std], dim=0))
        q_rescale_factor = compute_rescale_factor(torch.cat([training_female_mu, test_female_mu], dim=0), 
                                                     torch.cat([training_female_std, test_female_std], dim=0))
        
        p_training, q_training = generate_latent_image_data(training_male_mu, training_male_std,
                                                            training_female_mu, training_female_std,
                                                            p_rescale_factor, q_rescale_factor,
                                                            augmentation=True)
        p_test, q_test = generate_latent_image_data(test_male_mu, test_male_std,
                                                    test_female_mu, test_female_std,
                                                    p_rescale_factor, q_rescale_factor,
                                                    augmentation=False)
        
        # load celebA heldout image dataset
        p_test_dataset = CelebAValidMale(root=dataset_config.data_dir, size=64)
        q_test_dataset = CelebAValidFemale(root=dataset_config.data_dir, size=64)
        p_test_dataloader = DataLoader(ImageDefaultDataset(p_test_dataset), 
                                        batch_size=config.odeint_minibatch, 
                                        shuffle=False)
        q_test_dataloader = DataLoader(ImageDefaultDataset(q_test_dataset), 
                                        batch_size=config.odeint_minibatch, 
                                        shuffle=False)
        
    print("p_training size: {}".format(len(p_training)))
    print("q_training size: {}".format(len(q_training)))
    print("p_test size: {}".format(len(p_test)))
    print("q_test size: {}".format(len(q_test)))
        
    return {"p_training" : p_training, "q_training" : q_training,
            "p_test" : p_test, "q_test" : q_test,
            "p_rescale_factor" : p_rescale_factor, "q_rescale_factor" : q_rescale_factor,
            "p_test_dataloader" : p_test_dataloader, "q_test_dataloader" : q_test_dataloader,
            "image_dim" : image_dim}
    

def load_shoebags_latent_params(dataset_config):

    training_bags_latent_params = torch.load(dataset_config.train_encoded_bags_dir)
    training_shoes_latent_params = torch.load(dataset_config.train_encoded_shoes_dir)
    test_bags_latent_params = torch.load(dataset_config.test_encoded_bags_dir)
    test_shoes_latent_params = torch.load(dataset_config.test_encoded_shoes_dir)

    training_bags_mu, training_bags_logvar = torch.chunk(training_bags_latent_params, 2, dim=1) # (data_size, channel/2, w, h)
    training_shoes_mu, training_shoes_logvar = torch.chunk(training_shoes_latent_params, 2, dim=1) # (data_size, channel/2, w, h)
    test_bags_mu, test_bags_logvar = torch.chunk(test_bags_latent_params, 2, dim=1) # (data_size, channel/2, w, h)
    test_shoes_mu, test_shoes_logvar = torch.chunk(test_shoes_latent_params, 2, dim=1) # (data_size, channel/2, w, h)

    return (training_bags_mu, training_bags_logvar, training_shoes_mu, training_shoes_logvar, \
            test_bags_mu, test_bags_logvar, test_shoes_mu, test_shoes_logvar)


def load_celeba_latent_params(dataset_config):

    training_male_latent_params = torch.load(dataset_config.train_encoded_male_dir)
    training_female_latent_params = torch.load(dataset_config.train_encoded_female_dir)
    test_male_latent_params = torch.load(dataset_config.test_encoded_male_dir)
    test_female_latent_params = torch.load(dataset_config.test_encoded_female_dir)

    training_male_mu, training_male_logvar = torch.chunk(training_male_latent_params, 2, dim=1) # (data_size, channel/2, w, h)
    training_female_mu, training_female_logvar = torch.chunk(training_female_latent_params, 2, dim=1) # (data_size, channel/2, w, h)
    test_male_mu, test_male_logvar = torch.chunk(test_male_latent_params, 2, dim=1) # (data_size, channel/2, w, h)
    test_female_mu, test_female_logvar = torch.chunk(test_female_latent_params, 2, dim=1) # (data_size, channel/2, w, h)

    return (training_male_mu, training_male_logvar, training_female_mu, training_female_logvar, \
            test_male_mu, test_male_logvar, test_female_mu, test_female_logvar)


def compute_rescale_factor(mu, std):

    #total_sample = min(len(mu), 100000) # Use at most 100k samples to estimate the std
    total_sample = len(mu)
    z = (sample_latent_image(mu[:total_sample], std[:total_sample]))

    return z.flatten().std()


def sample_latent_image(mu, std, rescale=None):

    z = torch.randn_like(mu)

    if rescale:
        return (mu + std*z)/rescale  
    else:
        return mu + std*z
    

def generate_latent_image_data(p_mu, p_std, 
                               q_mu, q_std, 
                               p_rescale_factor, q_rescale_factor, 
                               augmentation=True):

    # compute multiplier to make the number of the generated latent images are the same between p and q
    if augmentation:
        if p_mu.size(0) > q_mu.size(0):
            p_multiplier = 1
            q_multiplier = (p_mu.size(0) // q_mu.size(0))+1
        else:
            p_multiplier = (q_mu.size(0) // p_mu.size(0))+1
            q_multiplier = 1
    else:
        p_multiplier = 1
        q_multiplier = 1


    generated_p_samples = []
    for _ in range(p_multiplier):
        generated_p_samples.append(sample_latent_image(p_mu, 
                                                        p_std, 
                                                        rescale=p_rescale_factor))

    generated_q_samples = []
    for _ in range(q_multiplier):
        generated_q_samples.append(sample_latent_image(q_mu, 
                                                        q_std,
                                                        rescale=q_rescale_factor))

        
    generated_p_samples = torch.cat(generated_p_samples, dim=0)
    generated_q_samples = torch.cat(generated_q_samples, dim=0)

    min_size = min(generated_p_samples.size(0), generated_q_samples.size(0))

    # both p and q samples have the same size 
    generated_p_samples = generated_p_samples[:min_size]
    generated_q_samples = generated_q_samples[:min_size]

    return generated_p_samples, generated_q_samples


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
    
