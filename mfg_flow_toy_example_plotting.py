from utils.plotting import plot_particle_trajectories_toy_example
import argparse


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--model_dir", type=str, default="./results/mfg_flow_toy_example/loop_1/velocity_field_l1.pt")
    p.add_argument("--config_dir", type=str, default="./results/mfg_flow_toy_example/mfg_flow_config.pkl")
    p.add_argument("--img_dir", type=str, default="./data/img_checkerboard_4x4.png")
    p.add_argument("--particle_trajectories_dir", 
                   type=str, default="./results/mfg_flow_toy_example/loop_1/optimized_particles_trajectories_l1.pt")
    p.add_argument("--num_selection", type=int, default=30)
    p.add_argument("--ode_solver", type=str, default="rk4")
    p.add_argument("--sample_size", type=int, default=20000)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--saving_dir", type=str, default="./results/mfg_flow_toy_example/loop_1/")

    return p.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    plot_particle_trajectories_toy_example(args.model_dir,
                                           args.config_dir,
                                           args.img_dir,
                                           args.particle_trajectories_dir,
                                           args.num_selection,
                                           args.ode_solver,
                                           args.sample_size,
                                           args.seed,
                                           args.saving_dir,
                                           )
    
    args = parse_args()
    args.model_dir = "./results/mfg_flow_toy_example/loop_10/velocity_field_l10.pt"
    args.particle_trajectories_dir = "./results/mfg_flow_toy_example/loop_10/optimized_particles_trajectories_l10.pt"
    args.saving_dir = "./results/mfg_flow_toy_example/loop_10/"
    
    plot_particle_trajectories_toy_example(args.model_dir,
                                           args.config_dir,
                                           args.img_dir,
                                           args.particle_trajectories_dir,
                                           args.num_selection,
                                           args.ode_solver,
                                           args.sample_size,
                                           args.seed,
                                           args.saving_dir,
                                           )