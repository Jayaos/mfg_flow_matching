from utils.plotting import plot_particle_trajectories_toy_example
import argparse


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--model_dir", type=str)
    p.add_argument("--config_dir", type=str)
    p.add_argument("--img_dir", type=str)
    p.add_argument("--particle_trajectories_dir", type=str)
    p.add_argument("--num_selection", type=int)
    p.add_argument("--ode_solver", type=str)
    p.add_argument("--sample_size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--saving_dir", type=str)

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