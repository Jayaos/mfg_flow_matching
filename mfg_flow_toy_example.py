from config import MFGFlowToyExampleConfig, ToyDatasetConfig
from src_run.run_2d import run_mfg_flow_toy_example
import torch
import argparse
import os


def parse_args():

    p = argparse.ArgumentParser()

    # data and problem
    p.add_argument("--saving_dir", type=str, default="./results/mfg_flow_toy_example/")
    p.add_argument("--data_dir", type=str, default="./data/")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n_train", type=int, default=100000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--outer_batch", type=int, default=10000)
    p.add_argument("--outer_loop", type=int, default=5)
    p.add_argument("--ode_solver", type=str, default="rk4")       
    p.add_argument("--odeint_minibatch", type=int, default=2048)               
    p.add_argument("--ode_timesteps", type=int, default=10)   

    # models
    p.add_argument("--classifier_hidden_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    p.add_argument("--vf_hidden_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    p.add_argument("--vf_layer_type", type=str, default="concatlinear")

    # training
    p.add_argument("--classifier_learning_rate", type=float, default=.001)
    p.add_argument("--classifier_minibatch", type=int, default=256)
    p.add_argument("--classifier_initial_steps", type=int, default=100)
    p.add_argument("--classifier_steps", type=int, default=100)
    p.add_argument("--cost_update_frequency", type=int, default=10)
    p.add_argument("--classifier_retrain_steps", type=int, default=20)

    p.add_argument("--vf_initialization", type=str, default=None)
    p.add_argument("--vf_initial_steps", type=int, default=500)
    p.add_argument("--vf_learning_rate", type=float, default=.001)  
    p.add_argument("--vf_minibatch", type=int, default=256)  
    p.add_argument("--vf_steps", type=int, default=500)  

    p.add_argument("--particle_learning_rate", type=float, default=.001)  
    p.add_argument("--particle_minibatch", type=int, default=256)
    p.add_argument("--particle_steps", type=int, default=500)        
    p.add_argument("--kinetic_loss_weight", type=float, default=0.05)
    p.add_argument("--classifier_loss_weight", type=float, default=1.0)

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
                                        args.n_train,
                                        args.n_test)
    

    q_dataset_config = ToyDatasetConfig("checkerboard-2d",
                                        os.path.join(args.data_dir, "img_checkerboard_4x4.png"),
                                        args.n_train,
                                        args.n_test)
    

    run_mfg_flow_toy_example(config, p_dataset_config, q_dataset_config, device=args.device)