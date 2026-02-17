from config import MFGFlowToyExampleConfig, ToyDatasetConfig
from src_run.run_2d import run_mfg_flow_toy_example
import torch
import argparse


def parse_args():

    p = argparse.ArgumentParser()

    # input/output
    p.add_argument("--saving_dir", type=str, default="./results/mfg_flow_toy_example/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)

    # classifier
    p.add_argument("--classifier_hidden_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    p.add_argument("--classifier_learning_rate", type=float, default=.001)
    p.add_argument("--classifier_training_batch_size", type=int, default=256)
    p.add_argument("--initial_classifier_training_step", type=int, default=100)
    p.add_argument("--classifier_intermediate_training_frequency", type=int, default=10)
    p.add_argument("--intermediate_classifier_training_step", type=int, default=20)

    # velocity field
    p.add_argument("--velocity_field_hidden_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    p.add_argument("--velocity_field_layer_type", type=str, default="concatlinear")
    p.add_argument("--velocity_field_learning_rate", type=float, default=.001)  
    p.add_argument("--velocity_field_training_batch_size", type=int, default=256)  
    p.add_argument("--velocity_field_initialization", type=str, default=None)
    p.add_argument("--velocity_field_initialization_training_step", type=int, default=500)
    p.add_argument("--initial_velocity_field_training_step", type=int, default=500)   
    p.add_argument("--velocity_field_training_step", type=int, default=500)  

    # particle optimization
    p.add_argument("--initial_particle_optimization_epoch", type=int, default=500)         
    p.add_argument("--particle_optimization_epoch", type=int, default=500)        
    p.add_argument("--particle_optimization_learning_rate", type=float, default=.001)  
    p.add_argument("--particle_optimization_batch_size", type=int, default=256)          

    # training
    p.add_argument("--kinetic_loss_weight", type=float, default=0.05)
    p.add_argument("--classifier_loss_weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--ode_solver", type=str, default="rk4")       
    p.add_argument("--odeint_batch_size", type=int, default=2048)               
    p.add_argument("--num_timesteps", type=int, default=10)   

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    config = MFGFlowToyExampleConfig(classifier_hidden_dims=args.classifier_hidden_dims,
                                     classifier_activation=torch.nn.ReLU(),
                                     classifier_learning_rate=args.classifier_learning_rate,
                                     classifier_training_batch_size=args.classifier_training_batch_size,
                                     initial_classifier_training_step=args.initial_classifier_training_step,
                                     classifier_intermediate_training_frequency=args.classifier_intermediate_training_frequency,
                                     intermediate_classifier_training_step=args.intermediate_classifier_training_step,
                                     velocity_field_hidden_dims=args.velocity_field_hidden_dims,
                                     velocity_field_layer_type=args.velocity_field_layer_type,
                                     velocity_field_activation = torch.nn.ReLU(),
                                     velocity_field_learning_rate=args.velocity_field_learning_rate,
                                     velocity_field_training_batch_size=args.velocity_field_training_batch_size,
                                     velocity_field_initialization=args.velocity_field_initialization,
                                     velocity_field_initialization_training_step=args.velocity_field_initialization_training_step,
                                     initial_velocity_field_training_step=args.initial_velocity_field_training_step,
                                     velocity_field_training_step=args.velocity_field_training_step,
                                     initial_particle_optimization_epoch=args.initial_particle_optimization_epoch,
                                     particle_optimization_epoch=args.particle_optimization_epoch,
                                     particle_optimization_learning_rate=args.particle_optimization_learning_rate,
                                     particle_optimization_batch_size=args.particle_optimization_batch_size,
                                     kinetic_loss_weight=args.kinetic_loss_weight,
                                     classifier_loss_weight=args.classifier_loss_weight,
                                     epochs=args.epochs,
                                     ode_solver=args.ode_solver,
                                     odeint_batch_size=args.odeint_batch_size,
                                     num_timesteps=args.num_timesteps,
                                     seed=args.seed,
                                     saving_dir=args.saving_dir)
        

    p_dataset_config = ToyDatasetConfig("Gaussian", 
                                        None,
                                        10000,
                                        2000)
    

    q_dataset_config = ToyDatasetConfig("checkerboard-2d", 
                                        "./data/img_checkerboard_4x4.png",
                                        10000,
                                        2000)
    

    run_mfg_flow_toy_example(config, p_dataset_config, q_dataset_config, device=args.device)