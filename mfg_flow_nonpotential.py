import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import MLPVelocityField, vf_loss_fn
from torchdiffeq import odeint
from scipy.stats import gaussian_kde


def run_mfg_nonpotential(num_outer_loop, particle_num, ode_timesteps, ode_solver,
                         particle_steps, step_size, cost_update_frequency, 
                         vf_hidden_dims, vf_steps, vf_learning_rate,
                         mu_0, cov_matrix_0, a, lamb_F, mu_1, lamb_G):
    """
    Args:
        num_outer_loop: Number of outer loop iterations
        particle_num: Number of particles
        ode_timesteps: Number of time steps for ODE
        ode_solver: ODE solver to use ("euler" or "rk4")
        particle_steps: Number of particle optimization steps
        step_size: Step size for particle optimization
        cost_update_frequency: Frequency of intermediate updates
        vf_hidden_dims: Hidden dimensions for neural network
        vf_steps: Number of velocity field training steps
        vf_learning_rate: Learning rate for velocity field
        mu_0: Initial mean
        cov_matrix_0: Initial covariance matrix
        a: Parameter vector
        lamb_F: Force parameter
        mu_1: Target mean
        lamb_G: Goal parameter
    
    Returns:
        velocity_field: Trained velocity field
        x_trajectory: Final particle trajectories
        results: Dictionary with results (residuals)
    """
    # setup
    initial_gaussian_dist = torch.distributions.MultivariateNormal(loc=mu_0, covariance_matrix=cov_matrix_0)  
    velocity_field = MLPVelocityField(2, 1, vf_hidden_dims, "concatlinear", activation=torch.nn.ReLU())
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=vf_learning_rate)
    timesteps = torch.linspace(0, 1, ode_timesteps)
    delta_t = 1/(ode_timesteps-1)

    residuals = []

    for k in range(num_outer_loop):

        print("outer loop {}".format(k+1))
        x_init = initial_gaussian_dist.sample([particle_num]) # (particle_num, dim)
        if k == 0:
            # initialization of trajectory: zero velocity field
            x_init = x_init.unsqueeze(1) # (particle_num, 1, dim)
            x_trajectory = x_init.repeat(1, ode_timesteps, 1) # (particle_num, num_timestep, dim)
            # # initialization of trajectory: random initialization
            # x_init = x_init.unsqueeze(1) # (particle_num, 1, dim)
            # x_trajectory = torch.randn(particle_num, num_timestep, 2) * 0.1 + mu_0 # (particle_num, num_timestep, dim)
            # x_trajectory[:,0,:] = x_init.squeeze(1)
            # # plot sample initialization trajectory
            # for i in range(10):
            #     plt.plot(x_trajectory[i,:,0].cpu().numpy(), x_trajectory[i,:,1].cpu().numpy(), '-o')
            # plt.title("Initialization of particle trajectories")
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.axis('equal')
            # plt.show()
        else:
            (particle_num, ode_timesteps, 2)
            with torch.no_grad():
                x_trajectory = odeint(velocity_field, x_init, timesteps, method=ode_solver)
            x_trajectory = x_trajectory.permute(1,0,2) # (particle_num, num_timestep, dim)

        # compute a dot x
        exp_term = torch.exp( torch.matmul(x_trajectory[:,1:-1,:], a) ).unsqueeze(-1) # (particle_num, num_timestep-1)
        mean_exp_term = torch.mean(exp_term, dim=0) # (num_timestep-1, )

        # particle optimization
        residuals_particle_optimization = []
        for j in range(particle_steps):
            with torch.no_grad():
                # (particle_num, num_timestep-2)
                DttX = (x_trajectory[:,:-2,:] - 2*x_trajectory[:,1:-1,:] + x_trajectory[:,2:,:]) * (1/delta_t**2) # pay attention
                FX = lamb_F * exp_term * a * mean_exp_term # (particle_num, num_timestep-2, dim)
                grad_int = delta_t * (-DttX + FX) # (particle_num, num_timestep-2, dim)

                DtX = (1/delta_t) * (x_trajectory[:,-1] - x_trajectory[:,-2]) # (particle_num, dim)
                GX = lamb_G * (x_trajectory[:,-1,:] - mu_1) # (particle_num, dim)
                GX[:,0] = 0 # only penalize y direction
                grad_term = DtX + GX # (particle_num, dim)

                # particle trajectory update
                x_trajectory[:,1:-1,:] += (-step_size * grad_int)
                x_trajectory[:,-1,:] += (-step_size * grad_term)

                if j == 0:
                    grad_mat = torch.cat([grad_int, grad_term.unsqueeze(1)], dim=1)
                    res = torch.norm(grad_mat, dim=(1,2)).mean()
    
                residuals_particle_optimization.append(res)

                if (j+1) % cost_update_frequency == 0:
                    # compute exp(a dot x) of the particle trajectory
                    exp_term = torch.exp( torch.matmul(x_trajectory[:,1:-1,:], a) ).unsqueeze(-1) # (particle_num, num_timestep-1)
                    mean_exp_term = torch.mean(exp_term, dim=0) # (num_timestep-1, )

        residuals.append(np.mean(residuals_particle_optimization))
        print("avg residual at outer loop {}: {}".format(k+1, np.mean(residuals_particle_optimization)))

        # update velocity field
        loss_sum = 0.
        for _ in range(vf_steps):

            loss = vf_loss_fn(velocity_field, x_trajectory, timesteps, delta_t)
            vf_optim.zero_grad()
            loss["loss"].backward()
            vf_optim.step()
            loss_sum += loss["loss"].item()

        print("avg vf loss at outer loop {}: {}".format(k+1, loss_sum/vf_steps))

        with torch.no_grad():
            x_trajectory_final = odeint(velocity_field, x_trajectory[:,0,:], timesteps, method=ode_solver)
        x_trajectory_final = x_trajectory_final.permute(1,0,2) # (particle_num, num_timestep)

        # mu_tj_record = torch.mean(x_trajectory_final, dim=0) # (num_timestep, )
        # mu_tj_error = delta_t*torch.linalg.vector_norm(mu_tj_record - mu_t_solution, ord=2)
        # mu_tj_errors.append(mu_tj_error.item())

    results = {"residuals" : residuals}
    
    return velocity_field, x_trajectory, results


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # data / problem
    parser.add_argument('--outer_loop', type=int, default=100,
                        help='Number of outer loop iterations')
    parser.add_argument('--n', type=int, default=1000,
                        help='Number of particles')
    parser.add_argument('--ode_timesteps', type=int, default=21,
                        help='Number of time steps')
    parser.add_argument('--ode_solver', type=str, default='euler',
                        choices=['euler', 'rk4'],
                        help='ODE solver to use')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--save_plots', type=str, default="./results/mfg_flow_nonpotential/",
                        help='Directory to save plots (if not specified, plots are displayed)')
    parser.add_argument('--outer_batch', type=int, default=5000,
                        help='Number of particles for resampling')
    
    # Non-potential game parameters (default values from cell 3)
    parser.add_argument('--mu_0', type=float, nargs=2, default=[0.0, 1.0],
                        help='Initial mean (2 values)')
    parser.add_argument('--cov_0_diag', type=float, nargs=2, default=[0.02, 0.1],
                        help='Diagonal of initial covariance matrix (2 values)')
    parser.add_argument('--a', type=float, nargs=2, default=[0.0, 1.0],
                        help='Parameter vector a (2 values)')
    parser.add_argument('--lamb_F', type=float, default=10.0,
                        help='Force parameter lambda_F')
    parser.add_argument('--mu_1', type=float, default=-1.0,
                        help='Target mean mu_1')
    parser.add_argument('--lamb_G', type=float, default=1.0,
                        help='Goal parameter lambda_G')
    
    # training
    parser.add_argument('--particle_steps', type=int, default=100,
                        help='Number of particle optimization steps')
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='step size for particle optimization')
    parser.add_argument('--cost_update_frequency', type=int, default=2,
                        help='cost update frequency')
    parser.add_argument('--vf_hidden_dims', type=int, nargs='+', default=[4, 8, 16],
                        help='Hidden dimensions for neural network')
    parser.add_argument('--vf_steps', type=int, default=100,
                        help='Number of velocity field training steps')
    parser.add_argument('--vf_learning_rate', type=float, default=0.01,
                        help='Velocity field learning rate')
    
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    print("=" * 60)
    print("Non-potential game with interaction cost defined by asymmetric kernel")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  outer loops: {args.outer_loop}")
    print(f"  n (particles): {args.n}")
    print(f"  m (timesteps): {args.ode_timesteps}")
    print(f"  ODE solver: {args.ode_solver}")
    print(f"  mu_0: {args.mu_0}")
    print(f"  cov_0_diag: {args.cov_0_diag}")
    print(f"  a: {args.a}")
    print(f"  lambda_F: {args.lamb_F}")
    print(f"  mu_1: {args.mu_1}")
    print(f"  lambda_G: {args.lamb_G}")
    print()

    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)
    
    # Convert arguments to tensors
    mu_0 = torch.tensor(args.mu_0, dtype=torch.float32)
    cov_matrix_0 = torch.diag(torch.tensor(args.cov_0_diag, dtype=torch.float32))
    a = torch.tensor(args.a, dtype=torch.float32)
    lamb_F = args.lamb_F
    mu_1 = args.mu_1
    lamb_G = args.lamb_G
    
    v, x, results = run_mfg_nonpotential(
        args.outer_loop, args.n, args.ode_timesteps, args.ode_solver,
        args.particle_steps, args.step_size, args.cost_update_frequency,
        args.vf_hidden_dims, args.vf_steps, args.vf_learning_rate,
        mu_0, cov_matrix_0, a, lamb_F, mu_1, lamb_G
    )
    print("Done!")
    
    residuals = results["residuals"]
    
    # Resample and compute trajectory
    print(f"\nResampling with {args.outer_batch} particles...")
    timesteps = torch.linspace(0, 1, args.ode_timesteps)
    initial_gaussian_dist = torch.distributions.MultivariateNormal(
        loc=mu_0, covariance_matrix=cov_matrix_0
    )
    x_init_resample = initial_gaussian_dist.sample([args.outer_batch])
    with torch.no_grad():
        x_resample = odeint(v, x_init_resample, timesteps, method=args.ode_solver)
    x_resample = x_resample.permute(1, 0, 2)  # (n_resample, num_timestep, dim)
    
    # Generate plots if not disabled
    if not args.no_plot:
        print("\nGenerating plots...")
        
        # Plot 1: Particle trajectories
        plt.figure(figsize=(5, 3))
        num_traj = 10
        colors = plt.cm.viridis(np.linspace(1, 0, args.ode_timesteps))
        sizes = np.linspace(20, 10, args.ode_timesteps)
        scatter_indices = [0, 5, 10, 15, 20] if args.ode_timesteps >= 21 else list(range(args.ode_timesteps))
        scatter_indices = [i for i in scatter_indices if i < args.ode_timesteps]
        
        for i in range(num_traj):
            plt.plot(x[i, :, 0].numpy(), x[i, :, 1].numpy(), color='#1f77b4', alpha=1)
            for j in scatter_indices:
                label = None
                if i == 0:
                    if j == 0:
                        label = 't=0'
                    elif j == args.ode_timesteps - 1:
                        label = 't=1'
                    else:
                        label = f't={timesteps[j]:.2f}'
                plt.scatter(x[i, j, 0].numpy(), x[i, j, 1].numpy(),
                           color=colors[j], s=sizes[j], label=label)
        
        plt.xlabel("x1")
        plt.ylabel("x2")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if args.save_plots:
            plt.savefig(f"{args.save_plots}/trajectories.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: {args.save_plots}/trajectories.png")
        else:
            plt.show()
        plt.close()
        
        # Plot 2: Residuals
        plt.figure(figsize=(4, 3))
        plt.plot(range(1, args.outer_loop + 1), residuals)
        plt.yscale("log")
        plt.xlabel("Epoch $k$")
        plt.ylabel("Residual")
        plt.grid()
        plt.tight_layout()
        
        if args.save_plots:
            plt.savefig(f"{args.save_plots}/res.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: {args.save_plots}/res.png")
        else:
            plt.show()
        plt.close()
        
        # Plot 3: Density evolution
        t_indices = np.linspace(0, args.ode_timesteps - 1, 8, dtype=int)
        fig, axes = plt.subplots(1, len(t_indices), figsize=(14, 2), sharex=True, sharey=True)
        region = [-0.6, 0.6, -1.2, 1.8]
        x1, x2 = np.mgrid[region[0]:region[1]:200j, region[2]:region[3]:200j]
        positions = np.vstack([x1.ravel(), x2.ravel()])
        
        for idx, t_idx in enumerate(t_indices):
            ax = axes[idx]
            data = x[:, t_idx, :].numpy().T
            kde = gaussian_kde(data)
            density = np.reshape(kde(positions).T, x1.shape)
            im = ax.imshow(np.rot90(density), cmap=plt.cm.viridis, extent=region, aspect='auto')
            ax.set_title(f"t={timesteps[t_idx]:.2f}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        if args.save_plots:
            plt.savefig(f"{args.save_plots}/density.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: {args.save_plots}/density.png")
        else:
            plt.show()
        plt.close()
        
        print("Plotting complete!")
    
    print("\n" + "=" * 60)
    print("Execution completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
