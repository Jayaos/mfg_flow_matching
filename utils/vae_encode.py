
from ldm.util import instantiate_from_config
from ldm.data.shoebags import *
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf




def vae_shoebags_encode(vae_config_dir, vae_model_dir, 
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

    shoes_train_dataset = Shoes(shoes_data_dir, train_ratio, 2025, False)
    shoes_test_dataset = Shoes(shoes_data_dir, train_ratio, 2025, True)
    bags_train_dataset = Bags(bags_data_dir, train_ratio, 2025, False)
    bags_test_dataset = Bags(bags_data_dir, train_ratio, 2025, True)

    shoes_train_dataloader = DataLoader(shoes_train_dataset, batch_size=batch_size, drop_last=False)
    shoes_test_dataloader = DataLoader(shoes_test_dataset, batch_size=batch_size, drop_last=False)
    bags_train_dataloader = DataLoader(bags_train_dataset, batch_size=batch_size, drop_last=False)
    bags_test_dataloader = DataLoader(bags_test_dataset, batch_size=batch_size, drop_last=False)

    # encode
    encoded = []
    for batch in tqdm(shoes_train_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "shoes_train_encoded.pt"))

    encoded = []
    for batch in tqdm(shoes_test_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "shoes_test_encoded.pt"))

    encoded = []
    for batch in tqdm(bags_train_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "bags_train_encoded.pt"))

    encoded = []
    for batch in tqdm(bags_test_dataloader):

        x = batch["image"].to(device)
        with torch.no_grad():
            h = model.encoder(x)
            params = model.quant_conv(h)
        encoded.append(params.cpu())

    encoded = torch.cat(encoded, dim=0)
    torch.save(encoded, os.path.join(saving_dir, "bags_test_encoded.pt"))


def vae_celeba_encode():
    ...