from ldm.util import instantiate_from_config
from ldm.data.shoebags import *
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/configs/vae_shoebags.yaml")

# Instantiate model from config
model = instantiate_from_config(config.model)

# Load weights
ckpt = torch.load("/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/logs/2025-08-20T02-31-30_vae_shoebags/checkpoints/last.ckpt", map_location='cuda:0')
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()
model.to("cuda:0")

shoes_dataset = Shoes("/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/data/shoes_64.hdf5",
                          0.9, 2025, False)

bags_dataset = Bags("/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/data/handbag_64.hdf5",
                         0.9, 2025, False)

shoes_dataloader = DataLoader(shoes_dataset, batch_size=128, drop_last=False)
bags_dataloader = DataLoader(bags_dataset, batch_size=128, drop_last=False)

# encode shoes

print("encoding shoes")
all_params = []
all_x = []
for batch in tqdm(shoes_dataloader):

    x = batch["image"].to("cuda:0")
    with torch.no_grad():
        h = model.encoder(x)
        params = model.quant_conv(h)
    all_params.append(params.cpu())
    all_x.append(x.cpu())
all_params = torch.cat(all_params, dim=0)
all_x = torch.cat(all_x, dim=0)

# save with matched order
torch.save(all_params, "/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/results/shoes_latent_params.pt")
torch.save(all_x, "/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/results/shoes_x.pt")

# encode bags

print("encoding bags")
all_params = []
all_x = []
for batch in tqdm(bags_dataloader):

    x = batch["image"].to("cuda:0")
    with torch.no_grad():
        h = model.encoder(x)
        params = model.quant_conv(h)
    all_params.append(params.cpu())
    all_x.append(x.cpu())
all_params = torch.cat(all_params, dim=0)
all_x = torch.cat(all_x, dim=0)

# save with matched order
torch.save(all_params, "/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/results/bags_latent_params.pt")
torch.save(all_x, "/storage/home/hcoda1/0/jlee3541/r-yxie77-0/projects/vae/results/bags_x.pt")

