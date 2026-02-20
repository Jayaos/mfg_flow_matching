from src_run.run_imgtranslation import run_mfg_flow_image
from config import ShoebagsDatasetConfig, MFGFlowImageConfig
import torch

import argparse


def parse_args():

    p = argparse.ArgumentParser()


    # input/output
    p.add_argument("--saving_dir", type=str, default="./results/mfg_flow_toy_example/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--vae_config_dir", type=str)
    p.add_argument("--vae_model_dir", type=str)


    # dataset
    p.add_argument("--shoes_dataset_dir", type=str)
    p.add_argument("--bags_dataset_dir", type=str)
    p.add_argument("--train_encoded_shoes_dir", type=str)
    p.add_argument("--train_encoded_bags_dir", type=str)
    p.add_argument("--test_encoded_shoes_dir", type=str)
    p.add_argument("--test_encoded_bags_dir", type=str)
    p.add_argument("--train_ratio", type=float, default=0.8)


    # classifier
    p.add_argument("--classifier_channels", type=int, nargs="+", default=[256, 512, 512, 1024, 1024])
    p.add_argument("--classifier_use_bias", type=bool, default=True)
    p.add_argument("--classifier_learning_rate", type=float, default=.001)
    p.add_argument("--classifier_training_batch_size", type=int, default=256)
    p.add_argument("--initial_classifier_training_step", type=int, default=2000)
    p.add_argument("--classifier_intermediate_training_frequency", type=int, default=10)
    p.add_argument("--intermediate_classifier_training_step", type=int, default=10)


    # velocity field
    p.add_argument("--velocity_field_encoding_dims", type=int, nargs="+", default=[64, 256, 512, 512, 1024])
    p.add_argument("--velocity_field_decoding_dims", type=int, nargs="+", default=[1024, 512, 512, 256, 64])
    p.add_argument("--velocity_field_kernel_sizes", type=int, nargs="+", default=[3,3,3,3,3,3,3,4,3,3])
    p.add_argument("--velocity_field_strides", type=int, nargs="+", default=[1,1,2,1,1,1,1,2,1,1])
    p.add_argument("--velocity_field_initialization", type=str, default="linear-flow-matching")
    p.add_argument("--velocity_field_initialization_training_step", type=int, default=10000)
    p.add_argument("--velocity_field_learning_rate", type=float, default=.001)  
    p.add_argument("--velocity_field_training_batch_size", type=int, default=256)  
    p.add_argument("--initial_velocity_field_training_step", type=int, default=1000)   
    p.add_argument("--velocity_field_training_step", type=int, default=1000)


    # particle optimization
    p.add_argument("--initial_particle_optimization_epoch", type=int, default=1000)         
    p.add_argument("--particle_optimization_epoch", type=int, default=1000)        
    p.add_argument("--particle_optimization_learning_rate", type=float, default=.001)  
    p.add_argument("--particle_optimization_batch_size", type=int, default=512)      


    # training
    p.add_argument("--kinetic_loss_weight", type=float, default=0.05)
    p.add_argument("--classifier_loss_weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--epoch_training_ratio", type=float, default=0.13)
    p.add_argument("--ode_solver", type=str, default="rk4")       
    p.add_argument("--odeint_batch_size", type=int, default=2048)               
    p.add_argument("--num_timesteps", type=int, default=15)   

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()


    dataset_config = ShoebagsDatasetConfig(args.shoes_dataset_dir,
                                           args.bags_dataset_dir,
                                           args.train_encoded_shoes_dir,
                                           args.train_encoded_bags_dir,
                                           args.test_encoded_shoes_dir,
                                           args.test_encoded_bags_dir,
                                           args.train_ratio,
                                           args.seed)


    config = MFGFlowImageConfig(classifier_channels = args.classifier_channels,
                    classifier_use_bias = args.classifier_use_bias,
                    classifier_learning_rate = args.classifier_learning_rate,
                    classifier_training_batch_size = args.classifier_training_batch_size,
                    initial_classifier_training_step = args.initial_classifier_training_step,
                    classifier_intermediate_training_frequency = args.classifier_intermediate_training_frequency,
                    intermediate_classifier_training_step = args.intermediate_classifier_training_step,
                    velocity_field_encoding_dims = args.velocity_field_encoding_dims,
                    velocity_field_decoding_dims = args.velocity_field_decoding_dims,
                    velocity_field_kernel_sizes = args.velocity_field_kernel_sizes,
                    velocity_field_strides = args.velocity_field_strides,
                    velocity_field_initialization = args.velocity_field_initialization,
                    velocity_field_initialization_training_step = args.velocity_field_initialization_training_step,
                    velocity_field_learning_rate = args.velocity_field_learning_rate,
                    velocity_field_training_batch_size = args.velocity_field_training_batch_size,
                    initial_velocity_field_training_step = args.initial_velocity_field_training_step,
                    velocity_field_training_step = args.velocity_field_training_step,
                    initial_particle_optimization_epoch = args.initial_particle_optimization_epoch,
                    particle_optimization_epoch = args.particle_optimization_epoch,
                    particle_optimization_learning_rate = args.particle_optimization_learning_rate,
                    particle_optimization_batch_size = args.particle_optimization_batch_size,
                    kinetic_loss_weight = args.kinetic_loss_weight,
                    classifier_loss_weight = args.classifier_loss_weight,
                    epochs = args.epochs,
                    epoch_training_ratio=args.epoch_training_ratio,
                    ode_solver = args.ode_solver,
                    odeint_batch_size = args.odeint_batch_size,
                    num_timesteps = args.num_timesteps,
                    vae_config_dir = args.vae_config_dir,
                    vae_model_dir = args.vae_model_dir,
                    saving_dir = args.saving_dir)


    run_mfg_flow_image(config, dataset_config, device=args.device)
