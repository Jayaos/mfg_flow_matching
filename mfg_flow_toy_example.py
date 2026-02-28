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
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n_train", type=int, default=500000)
    p.add_argument("--n_test", type=int, default=20000)
    p.add_argument("--outer_batch", type=int, default=20000)
    p.add_argument("--outer_loop", type=int, default=20)
    p.add_argument("--ode_solver", type=str, default="rk4")       
    p.add_argument("--odeint_minibatch", type=int, default=2048)               
    p.add_argument("--ode_timesteps", type=int, default=10)   

    # models
    p.add_argument("--classifier_hidden_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    p.add_argument("--vf_hidden_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    p.add_argument("--vf_layer_type", type=str, default="concatlinear")

    # training
    p.add_argument("--classifier_learning_rate", type=float, default=.001)
    p.add_argument("--classifier_minibatch", type=int, default=2048)
    p.add_argument("--classifier_initial_steps", type=int, default=1000)
    p.add_argument("--cost_update_frequency", type=int, default=10)
    p.add_argument("--classifier_retrain_steps", type=int, default=10)

    p.add_argument("--vf_initialization", type=str, default=None)
    p.add_argument("--vf_initial_steps", type=int, default=1000)
    p.add_argument("--vf_learning_rate", type=float, default=.001)  
    p.add_argument("--vf_minibatch", type=int, default=2048)  
    p.add_argument("--vf_steps", type=int, default=100)  

    p.add_argument("--particle_learning_rate", type=float, default=.001)  
    p.add_argument("--particle_minibatch", type=int, default=2048)
    p.add_argument("--particle_loop", type=int, default=100)        
    p.add_argument("--kinetic_loss_weight", type=float, default=0.05)
    p.add_argument("--classifier_loss_weight", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    config = MFGFlowToyExampleConfig(saving_dir=args.saving_dir,
                                     data_dir=args.data_dir,
                                     seed=args.seed,
                                     n_train=args.n_train,
                                     n_test=args.n_test,
                                     outer_batch=args.outer_batch,
                                     outer_loop=args.outer_loop,
                                     ode_solver=args.ode_solver,
                                     odeint_minibatch=args.odeint_minibatch,
                                     ode_timesteps=args.ode_timesteps,

                                    classifier_hidden_dims=args.classifier_hidden_dims,
                                    classifier_activation=torch.nn.ReLU(),
                                    vf_hidden_dims=args.vf_hidden_dims,      
                                    vf_layer_type=args.vf_layer_type,           
                                    vf_activation=torch.nn.ReLU(),

                                    classifier_learning_rate=args.classifier_learning_rate,
                                    classifier_minibatch=args.classifier_minibatch,
                                    classifier_initial_steps=args.classifier_initial_steps,
                                    cost_update_frequency=args.cost_update_frequency,
                                    classifier_retrain_steps=args.classifier_retrain_steps,

                                    vf_initialization=args.vf_initialization,
                                    vf_initial_steps=args.vf_initial_steps,
                                    vf_learning_rate=args.vf_learning_rate,     
                                    vf_minibatch=args.vf_minibatch,
                                    vf_steps=args.vf_steps,

                                    particle_learning_rate=args.particle_learning_rate,
                                    particle_minibatch=args.particle_minibatch,
                                    particle_loop=args.particle_loop,
                                    kinetic_loss_weight=args.kinetic_loss_weight,
                                    classifier_loss_weight=args.classifier_loss_weight)
    
    
    p_dataset_config = ToyDatasetConfig("checkerboard-2d",
                                        os.path.join(args.data_dir, "img_checkerboard_4x4.png"),
                                        args.n_train,
                                        args.n_test)
    
        
    q_dataset_config = ToyDatasetConfig("Gaussian", 
                                        None,
                                        args.n_train,
                                        args.n_test)
 

    run_mfg_flow_toy_example(config, p_dataset_config, q_dataset_config, device=args.device)