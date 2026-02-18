from utils.vae_encode import vae_encode_celeba
import argparse
import torch


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--vae_config_dir", type=str)
    p.add_argument("--vae_model_dir", type=str)
    p.add_argument("--celeba_data_dir", type=str)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--saving_dir", type=str)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    vae_encode_celeba(args.vae_config_dir,
                      args.vae_model_dir, 
                      args.celeba_data_dir,
                      args.batch_size,
                      args.saving_dir,
                      args.device)