from src_run.run_imgtranslation import run_mfg_flow_image
from config import ShoebagsDatasetConfig, MFGFlowImageConfig
import torch
import sys
import argparse


def parse_args():

    p = argparse.ArgumentParser()

    # data / problem
    p.add_argument("--saving_dir", type=str, default="./results/mfg_flow_toy_example/")
    p.add_argument("--vae_config_dir", type=str, default="./vae_mfg_flow_matching/vae_shoebags_config.yaml")
    p.add_argument("--vae_model_dir", type=str, default="./vae_mfg_flow_matching/vae_shoebags.ckpt")
    p.add_argument("--shoes_data_dir", type=str, default="./data/shoes_64.hdf5")
    p.add_argument("--bags_data_dir", type=str, default="./data/handbag_64.hdf5")
    p.add_argument("--train_encoded_shoes_dir", type=str, default="./data/train_encoded_shoes.pt")
    p.add_argument("--train_encoded_bags_dir", type=str, default="./data/train_encoded_bags.pt")
    p.add_argument("--test_encoded_shoes_dir", type=str, default="./data/test_encoded_shoes.pt")
    p.add_argument("--test_encoded_bags_dir", type=str,  default="./data/test_encoded_bags.pt")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--outer_batch", type=int, default=10000)
    p.add_argument("--outer_loop", type=int, default=50)
    p.add_argument("--ode_solver", type=str, default="rk4")       
    p.add_argument("--odeint_minibatch", type=int, default=1024)
    p.add_argument("--ode_timesteps", type=int, default=15)

    # models
    p.add_argument("--classifier_channels", type=int, nargs="+", default=[256, 512, 512, 1024, 1024])
    p.add_argument("--classifier_use_bias", type=bool, default=True)
    p.add_argument("--vf_encoding_dims", type=int, nargs="+", default=[64, 256, 512, 512, 1024])
    p.add_argument("--vf_decoding_dims", type=int, nargs="+", default=[1024, 512, 512, 256, 64])
    p.add_argument("--vf_kernel_sizes", type=int, nargs="+", default=[3,3,3,3,3,3,3,4,3,3])
    p.add_argument("--vf_strides", type=int, nargs="+", default=[1,1,2,1,1,1,1,2,1,1])

    
    # training
    p.add_argument("--classifier_learning_rate", type=float, default=.001)
    p.add_argument("--classifier_minibatch", type=int, default=256)
    p.add_argument("--classifier_initial_steps", type=int, default=2000)
    p.add_argument("--cost_update_frequency", type=int, default=10)
    p.add_argument("--classifier_retrain_steps", type=int, default=10)

    p.add_argument("--vf_initialization", type=str, default="linear-flow-matching")
    p.add_argument("--vf_initial_steps", type=int, default=10000)
    p.add_argument("--vf_learning_rate", type=float, default=.001)  
    p.add_argument("--vf_minibatch", type=int, default=256)  
    p.add_argument("--vf_steps", type=int, default=1000)  

    p.add_argument("--particle_learning_rate", type=float, default=.001)  
    p.add_argument("--particle_minibatch", type=int, default=512)
    p.add_argument("--particle_loop", type=int, default=1000)        
    p.add_argument("--kinetic_loss_weight", type=float, default=0.05)
    p.add_argument("--classifier_loss_weight", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    args = parse_args()


    dataset_config = ShoebagsDatasetConfig(args.shoes_data_dir,
                                           args.bags_data_dir,
                                           args.train_encoded_shoes_dir,
                                           args.train_encoded_bags_dir,
                                           args.test_encoded_shoes_dir,
                                           args.test_encoded_bags_dir,
                                           args.train_ratio,
                                           args.seed)


    config = MFGFlowImageConfig(classifier_channels = args.classifier_channels,
                    classifier_use_bias = args.classifier_use_bias,
                    classifier_learning_rate = args.classifier_learning_rate,
                    classifier_minibatch = args.classifier_minibatch,
                    classifier_initial_steps = args.classifier_initial_steps,
                    cost_update_frequency = args.cost_update_frequency,
                    classifier_retrain_steps = args.classifier_retrain_steps,
                    vf_encoding_dims = args.vf_encoding_dims,
                    vf_decoding_dims = args.vf_decoding_dims,
                    vf_kernel_sizes = args.vf_kernel_sizes,
                    vf_strides = args.vf_strides,
                    vf_initialization = args.vf_initialization,
                    vf_initial_steps = args.vf_initial_steps,
                    vf_learning_rate = args.vf_learning_rate,
                    vf_minibatch = args.vf_minibatch,
                    vf_steps = args.vf_steps,
                    particle_loop = args.particle_loop,
                    particle_learning_rate = args.particle_learning_rate,
                    particle_minibatch = args.particle_minibatch,
                    kinetic_loss_weight = args.kinetic_loss_weight,
                    classifier_loss_weight = args.classifier_loss_weight,
                    outer_loop = args.outer_loop,
                    outer_batch=args.outer_batch,
                    ode_solver = args.ode_solver,
                    odeint_minibatch = args.odeint_minibatch,
                    ode_timesteps = args.ode_timesteps,
                    vae_config_dir = args.vae_config_dir,
                    vae_model_dir = args.vae_model_dir,
                    saving_dir = args.saving_dir)


    run_mfg_flow_image(config, dataset_config, device=args.device)
