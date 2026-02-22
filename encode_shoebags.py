from utils.vae_encode import vae_encode_shoebags
import argparse
import torch


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--vae_config_dir", type=str, default="./vae_mfg_flow_matching/vae_shoebags_config.yaml")
    p.add_argument("--vae_model_dir", type=str, default="./vae_mfg_flow_matching/vae_shoebags.ckpt")
    p.add_argument("--shoes_data_dir", type=str, default="./data/shoes_64.hdf5")
    p.add_argument("--bags_data_dir", type=str, default="./data/handbag_64.hdf5")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--saving_dir", type=str, default="./data/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    vae_encode_shoebags(args.vae_config_dir,
                        args.vae_model_dir, 
                        args.shoes_data_dir,
                        args.bags_data_dir,
                        args.train_ratio,
                        args.seed,
                        args.batch_size,
                        args.saving_dir,
                        args.device)