import torch
from torch.utils.data import DataLoader, TensorDataset
from ldm.util import instantiate_from_config
from ldm.data.base import ImageDefaultDataset
from ldm.data.shoebags import Shoes, Bags
from ldm.data.celeba import CelebAValidMale, CelebAValidFemale, CelebATestMale, CelebATestFemale
from evaluation.fid.eval import get_inception_features, build_reconstructed_data, compute_frechet_distance
from omegaconf import OmegaConf


def run_evaluate_vae_fid_shoebags(vae_config_dir, 
                                  vae_model_dir, 
                                  shoes_dataset_dir, 
                                  bags_dataset_dir, 
                                  dataset_frac, 
                                  batch_size, 
                                  seed, 
                                  device):
    """
    evaluate the trained VAE by computing FID score
    option 1: training samples vs. training samples
    option 2: held-out test samples vs. held-out test samples
    option 3: training samples vs. held-out test samples

    Args
    ----
        dataset_frac: train size frac
        seed
        use_heldout: bool, use held-out dataset for evaluation if True

    Return
    ------

    """

    # load trained VAE
    print("loading pre-trained autoencoder...")
    vae_config = OmegaConf.load(vae_config_dir)
    vae_model = instantiate_from_config(vae_config.model)
    vae_model.load_state_dict(torch.load(vae_model_dir)["state_dict"], strict=False)

    # load image dataset
    shoes_training_dataset = Shoes(shoes_dataset_dir, dataset_frac, seed, complement=False)
    bags_training_dataset = Bags(bags_dataset_dir, dataset_frac, seed, complement=False)
    shoes_test_dataset = Shoes(shoes_dataset_dir, dataset_frac, seed, complement=True)
    bags_test_dataset = Bags(bags_dataset_dir, dataset_frac, seed, complement=True)

    shoes_training_dataloader = DataLoader(ImageDefaultDataset(shoes_training_dataset), 
                                          batch_size=batch_size, 
                                          shuffle=False)
    shoes_test_dataloader = DataLoader(ImageDefaultDataset(shoes_test_dataset), 
                                       batch_size=batch_size, 
                                       shuffle=False)
    bags_training_dataloader = DataLoader(ImageDefaultDataset(bags_training_dataset), 
                                          batch_size=batch_size, 
                                          shuffle=False)
    bags_test_dataloader = DataLoader(ImageDefaultDataset(bags_test_dataset), 
                                       batch_size=batch_size, 
                                       shuffle=False)
    
    # reconstruct using VAE
    print("building reconstructed datasets using VAE...")
    shoes_training_reconstructed = build_reconstructed_data(vae_model, shoes_training_dataloader, device=device)
    shoes_test_reconstructed = build_reconstructed_data(vae_model, shoes_test_dataloader, device=device)
    bags_training_reconstructed = build_reconstructed_data(vae_model, bags_training_dataloader, device=device)
    bags_test_reconstructed = build_reconstructed_data(vae_model, bags_test_dataloader, device=device)

    # compute inception features from reconstructed datasets
    shoes_training_rec_mu, shoes_training_rec_sigma = get_inception_features(DataLoader(TensorDataset(shoes_training_reconstructed), batch_size=batch_size),
                           dims=2048,
                           device=device)
    shoes_test_rec_mu, shoes_test_rec_sigma = get_inception_features(DataLoader(TensorDataset(shoes_test_reconstructed), batch_size=batch_size),
                           dims=2048,
                           device=device)
    bags_training_rec_mu, bags_training_rec_sigma = get_inception_features(DataLoader(TensorDataset(bags_training_reconstructed), batch_size=batch_size),
                           dims=2048,
                           device=device)
    bags_test_rec_mu, bags_test_rec_sigma = get_inception_features(DataLoader(TensorDataset(bags_test_reconstructed), batch_size=batch_size),
                           dims=2048,
                           device=device)
    
    # compute inception features from images
    shoes_training_true_mu, shoes_training_true_sigma = get_inception_features(shoes_training_dataloader, 
                                                                               dims=2048, 
                                                                               device=device)
    shoes_test_true_mu, shoes_test_true_sigma = get_inception_features(shoes_test_dataloader, 
                                                                               dims=2048, 
                                                                               device=device)
    bags_training_true_mu, bags_training_true_sigma = get_inception_features(bags_training_dataloader, 
                                                                               dims=2048, 
                                                                               device=device)
    bags_test_true_mu, bags_test_true_sigma = get_inception_features(bags_test_dataloader, 
                                                                     dims=2048, 
                                                                     device=device)

    # conpute FID
    shoes_training_fid = compute_frechet_distance(shoes_training_rec_mu, 
                                                  shoes_training_rec_sigma, 
                                                  shoes_training_true_mu, 
                                                  shoes_training_true_sigma)
    shoes_test_fid = compute_frechet_distance(shoes_test_rec_mu, 
                                                  shoes_test_rec_sigma, 
                                                  shoes_test_true_mu, 
                                                  shoes_test_true_sigma)
    bags_training_fid = compute_frechet_distance(bags_training_rec_mu, 
                                                bags_training_rec_sigma, 
                                                bags_training_true_mu, 
                                                bags_training_true_sigma)
    bags_test_fid = compute_frechet_distance(bags_test_rec_mu, 
                                             bags_test_rec_sigma, 
                                             bags_test_true_mu, 
                                             bags_test_true_sigma)
    
    shoes_training_test_fid = compute_frechet_distance(shoes_training_rec_mu, 
                                                  shoes_training_rec_sigma, 
                                                  shoes_test_true_mu, 
                                                  shoes_test_true_sigma)
    
    bags_training_test_fid = compute_frechet_distance(bags_training_rec_mu, 
                                                  bags_training_rec_sigma, 
                                                  bags_test_true_mu, 
                                                  bags_test_true_sigma)
    
    print("shoes training FID: {}".format(shoes_training_fid))
    print("shoes test FID: {}".format(shoes_test_fid))
    print("bags training FID: {}".format(bags_training_fid))
    print("bags test FID: {}".format(bags_test_fid))
    print("shoes training vs. test FID: {}".format(shoes_training_test_fid))
    print("bags training vs. test FID: {}".format(bags_training_test_fid))


def run_evaluate_vae_fid_celeba(vae_config_dir, 
                                vae_model_dir, 
                                data_dir, 
                                batch_size, 
                                device):
    """
    evaluate the trained VAE by computing FID score
    option 1: training samples vs. training samples
    option 2: held-out test samples vs. held-out test samples
    option 3: training samples vs. held-out test samples

    Args
    ----
        dataset_frac: train size frac
        seed
        use_heldout: bool, use held-out dataset for evaluation if True

    Return
    ------

    """
    # load trained VAE
    print("loading pre-trained autoencoder...")
    vae_config = OmegaConf.load(vae_config_dir)
    vae_model = instantiate_from_config(vae_config.model)
    vae_model.load_state_dict(torch.load(vae_model_dir)["state_dict"], strict=False)

    # load image dataset
    # valid set is test set in our setting
    celeba_male_test_data = CelebAValidMale(root=data_dir, size=64)
    celeba_female_test_data = CelebAValidFemale(root=data_dir, size=64)
    celeba_male_heldout_data = CelebATestMale(root=data_dir, size=64)
    celeba_female_heldout_data = CelebATestFemale(root=data_dir, size=64)

    celeba_male_test_dataloader = DataLoader(ImageDefaultDataset(celeba_male_test_data), 
                                          batch_size=batch_size, 
                                          shuffle=False)
    celeba_female_test_dataloader = DataLoader(ImageDefaultDataset(celeba_female_test_data), 
                                          batch_size=batch_size, 
                                          shuffle=False)
    celeba_male_heldout_dataloader = DataLoader(ImageDefaultDataset(celeba_male_heldout_data), 
                                       batch_size=batch_size, 
                                       shuffle=False)
    celeba_female_heldout_dataloader = DataLoader(ImageDefaultDataset(celeba_female_heldout_data), 
                                       batch_size=batch_size, 
                                       shuffle=False)
    
    # reconstruct using VAE
    print("building reconstructed datasets using VAE...")
    celeba_male_test_rec = build_reconstructed_data(vae_model, celeba_male_test_dataloader, device=device)
    celeba_female_test_rec = build_reconstructed_data(vae_model, celeba_female_test_dataloader, device=device)
    celeba_male_heldout_rec = build_reconstructed_data(vae_model, celeba_male_heldout_dataloader, device=device)
    celeba_female_heldout_rec = build_reconstructed_data(vae_model, celeba_female_heldout_dataloader, device=device)

    # compute inception features from reconstructed datasets
    celeba_male_test_rec_mu, celeba_male_test_rec_sigma = get_inception_features(DataLoader(
                                                            TensorDataset(celeba_male_test_rec), 
                                                            batch_size=batch_size),
                                                            dims=2048,
                                                            device=device)
    celeba_female_test_rec_mu, celeba_female_test_rec_sigma = get_inception_features(DataLoader(
                                                            TensorDataset(celeba_female_test_rec), 
                                                            batch_size=batch_size),
                                                            dims=2048,
                                                            device=device)
    celeba_male_heldout_rec_mu, celeba_male_heldout_rec_sigma = get_inception_features(DataLoader(
                                                            TensorDataset(celeba_male_heldout_rec), 
                                                            batch_size=batch_size),
                                                            dims=2048,
                                                            device=device)
    celeba_female_heldout_rec_mu, celeba_female_heldout_rec_sigma = get_inception_features(DataLoader(TensorDataset(
                                                            celeba_female_heldout_rec), 
                                                            batch_size=batch_size),
                                                            dims=2048,
                                                            device=device)
    
    # compute inception features from images
    celeba_male_test_mu, celeba_male_test_sigma = get_inception_features(celeba_male_test_dataloader, 
                                                                                   dims=2048, 
                                                                                   device=device)
    celeba_female_test_mu, celeba_female_test_sigma = get_inception_features(celeba_female_test_dataloader, 
                                                                               dims=2048, 
                                                                               device=device)
    celeba_male_heldout_mu, celeba_male_heldout_sigma = get_inception_features(celeba_male_heldout_dataloader, 
                                                                               dims=2048, 
                                                                               device=device)
    celeba_female_heldout_mu, celeba_female_heldout_sigma = get_inception_features(celeba_female_heldout_dataloader, 
                                                                     dims=2048, 
                                                                     device=device)

    # conpute FID
    celeba_male_test_rec_fid = compute_frechet_distance(celeba_male_test_rec_mu, 
                                                        celeba_male_test_rec_sigma,
                                                        celeba_male_test_mu,
                                                        celeba_male_test_sigma)
    celeba_male_heldout_rec_fid = compute_frechet_distance(celeba_male_heldout_rec_mu, 
                                                           celeba_male_heldout_rec_sigma,
                                                           celeba_male_heldout_mu,
                                                           celeba_male_heldout_sigma)
    celeba_female_test_rec_fid = compute_frechet_distance(celeba_female_test_rec_mu, 
                                                        celeba_female_test_rec_sigma,
                                                        celeba_female_test_mu,
                                                        celeba_female_test_sigma)
    celeba_female_heldout_rec_fid = compute_frechet_distance(celeba_female_heldout_rec_mu, 
                                                           celeba_female_heldout_rec_sigma,
                                                           celeba_female_heldout_mu,
                                                           celeba_female_heldout_sigma)
    celeba_male_test_rec_heldout_fid = compute_frechet_distance(celeba_male_test_rec_mu, 
                                                        celeba_male_test_rec_sigma,
                                                        celeba_male_heldout_mu,
                                                        celeba_male_heldout_sigma)
    celeba_female_test_rec_heldout_fid = compute_frechet_distance(celeba_female_test_rec_mu, 
                                                        celeba_female_test_rec_sigma,
                                                        celeba_female_heldout_mu,
                                                        celeba_female_heldout_sigma)
    
    print("celebA male reconstruction vs. raw images FID on test set: {}".format(celeba_male_test_rec_fid))
    print("celebA male reconstruction vs. raw images FID on held-out set: {}".format(celeba_male_heldout_rec_fid))
    print("celebA female reconstruction vs. raw images FID on test set: {}".format(celeba_female_test_rec_fid))
    print("celebA female reconstruction vs. raw images FID on held-out set: {}".format(celeba_female_heldout_rec_fid))
    print("celebA male test reconstruction vs. held-out raw images FID on test set: {}".format(celeba_male_test_rec_heldout_fid))
    print("celebA female test reconstruction vs. held-out raw images FID on test set: {}".format(celeba_female_test_rec_heldout_fid))
