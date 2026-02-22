
from ldm.util import instantiate_from_config
from ldm.data.shoebags import *
from ldm.data.celeba import *
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


def vae_encode_shoebags(vae_config_dir, vae_model_dir, 
                        shoes_data_dir, bags_data_dir, train_ratio, seed,
                        batch_size, saving_dir, device):

    # Load config
    config = OmegaConf.load(vae_config_dir)

    # Instantiate model from config
    model = instantiate_from_config(config.model)

    # Load weights
    ckpt = torch.load(vae_model_dir, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)

    shoes_train_dataset = Shoes(shoes_data_dir, train_ratio, seed, False)
    shoes_test_dataset = Shoes(shoes_data_dir, train_ratio, seed, True)
    bags_train_dataset = Bags(bags_data_dir, train_ratio, seed, False)
    bags_test_dataset = Bags(bags_data_dir, train_ratio, seed, True)

    shoes_train_dataloader = DataLoader(shoes_train_dataset, batch_size=batch_size, drop_last=False)
    shoes_test_dataloader = DataLoader(shoes_test_dataset, batch_size=batch_size, drop_last=False)
    bags_train_dataloader = DataLoader(bags_train_dataset, batch_size=batch_size, drop_last=False)
    bags_test_dataloader = DataLoader(bags_test_dataset, batch_size=batch_size, drop_last=False)

    # encode
    print("encoding shoes for train dataset")
    encoded = []
    for batch in tqdm(shoes_train_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "train_encoded_shoes.pt"))

    print("encoding shoes for test dataset")
    encoded = []
    for batch in tqdm(shoes_test_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "test_encoded_shoes.pt"))

    print("encoding bags for train dataset")
    encoded = []
    for batch in tqdm(bags_train_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "train_encoded_bags.pt"))

    print("encoding bags for test dataset")
    encoded = []
    for batch in tqdm(bags_test_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "test_encoded_bags.pt"))


def vae_encode_celeba(vae_config_dir, vae_model_dir, celeba_data_dir, batch_size, saving_dir, device):

    # Load config
    config = OmegaConf.load(vae_config_dir)

    # Instantiate model from config
    model = instantiate_from_config(config.model)

    # Load weights
    ckpt = torch.load(vae_model_dir, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)

    celeba_male_train = CelebATrainMale(root=celeba_data_dir, size=64)
    celeba_female_train = CelebATrainFemale(root=celeba_data_dir, size=64)
    celeba_male_valid = CelebAValidMale(root=celeba_data_dir, size=64)
    celeba_female_valid = CelebAValidFemale(root=celeba_data_dir, size=64)
    celeba_male_test = CelebATestMale(root=celeba_data_dir, size=64)
    celeba_female_test = CelebATestFemale(root=celeba_data_dir, size=64)

    celeba_male_train_dataloader = DataLoader(celeba_male_train, batch_size=batch_size, drop_last=False)
    celeba_female_train_dataloader = DataLoader(celeba_female_train, batch_size=batch_size, drop_last=False)
    celeba_male_valid_dataloader = DataLoader(celeba_male_valid, batch_size=batch_size, drop_last=False)
    celeba_female_valid_dataloader = DataLoader(celeba_female_valid, batch_size=batch_size, drop_last=False)
    celeba_male_test_dataloader = DataLoader(celeba_male_test, batch_size=batch_size, drop_last=False)
    celeba_female_test_dataloader = DataLoader(celeba_female_test, batch_size=batch_size, drop_last=False)

    # encode
    # train and valid set into a single training set
    print("encoding CelebA male for train dataset")
    encoded = []
    for batch in tqdm(celeba_male_train_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    for batch in tqdm(celeba_male_valid_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "train_encoded_male.pt"))

    print("encoding CelebA female for train dataset")
    encoded = []
    for batch in tqdm(celeba_female_train_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    for batch in tqdm(celeba_female_valid_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "train_encoded_female.pt"))

    print("encoding CelebA male for test dataset")
    encoded = []
    for batch in tqdm(celeba_male_test_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "test_encoded_male.pt"))

    print("encoding CelebA female for test dataset")
    encoded = []
    for batch in tqdm(celeba_female_test_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "test_encoded_female.pt"))

