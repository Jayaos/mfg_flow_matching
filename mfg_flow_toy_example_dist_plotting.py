from utils.utils import load_data
from utils.data import generate_toy_data
from utils.plotting import plot_2d_ode_trajectories
from model import MLPVelocityField
import torch
import argparse


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--p_dataset_config_dir", type=str, default="./results/mfg_flow_toy_example/p_dataset_config.pkl")
    p.add_argument("--q_dataset_config_dir", type=str, default="./results/mfg_flow_toy_example/q_dataset_config.pkl")
    p.add_argument("--model_dir", type=str)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--config_dir", type=str, default="./results/mfg_flow_toy_example/mfg_flow_config.pkl")
    p.add_argument("--sample_size", type=int, default=30000)
    p.add_argument("--saving_dir", type=str)

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    p_dataset_config = load_data(args.p_dataset_config_dir)
    q_dataset_config = load_data(args.q_dataset_config_dir)
    p_dataset_config.n_test = args.sample_size
    q_dataset_config.n_test = args.sample_size

    _, p_test = generate_toy_data(p_dataset_config, args.seed)
    _, q_test = generate_toy_data(q_dataset_config, args.seed)

    config = load_data(args.config_dir)

    velocity_field = MLPVelocityField(
        2,
        1,  # time dim = 1
        config.vf_hidden_dims,
        config.vf_layer_type,
        config.vf_activation,
    )
    velocity_field.load_state_dict(torch.load(args.model_dir, map_location=torch.device("cpu")))

    plot_2d_ode_trajectories(velocity_field, 
                             p_test, q_test, 
                             config.ode_timesteps, 
                             8, 200, config.ode_solver, 
                             config.odeint_minibatch, 
                             device="cpu", 
                             saving=args.saving_dir)
