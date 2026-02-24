import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import os
from tqdm import tqdm
from torchdiffeq import odeint
from .data import generate_checkerboard_2d
from model import MLPVelocityField
from .utils import load_data, save_data
from .utils import batched_odeint
from scipy.stats import gaussian_kde


def plot_loss_record_epoch(loss_record_epoch):

    fig = plt.figure(figsize=(30, 7))  # Increased height
    # Create a grid for the top row (3 columns for losses)
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    # Plot losses
    fsize = 16
    ax1.plot(loss_record_epoch["kinetic_loss_record"])
    ax1.set_title("kinetic_loss_record", fontsize=fsize)
    ax2.plot(loss_record_epoch["classifier_loss_record"])
    ax2.set_title("classifier_loss_record", fontsize=fsize)
    ax3.plot(loss_record_epoch["particle_optimization_loss_record"])
    ax3.set_title("particle_optimization_loss_record", fontsize=fsize)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Batch', fontsize=fsize)
        ax.set_ylabel('Loss', fontsize=fsize)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.35)  # Increase vertical space between subplots


def plot_particles(particle_trajectories, num_plots, device='cpu'):
    num_timesteps = particle_trajectories.size(1)
    max_plot = min(num_plots, num_timesteps)
    timesteps = torch.linspace(0, 1, num_timesteps)

    fig = plt.figure(figsize=(3 * max_plot, 3))  # single row height

    Xbar_plt = particle_trajectories.detach().cpu().numpy()
    idx = torch.linspace(0, num_timesteps - 1, max_plot).long().to(device)

    for i in range(max_plot):
        ax = fig.add_subplot(1, max_plot, i + 1)  # single row
        ax.scatter(Xbar_plt[:, idx[i], 0], Xbar_plt[:, idx[i], 1], s=1, alpha=0.5)
        ax.set_title(f't={timesteps[idx[i]]:.2f}')

    plt.tight_layout()
    plt.show()


def plot_random_trajectory(model_dir,
                           config_dir,
                           img_dir,
                           particle_trajectories_dir,
                           particle_trajectories_prior_dir,
                           num_selection,
                           ode_solver,
                           sample_size,
                           seed=None):

    config = load_data(config_dir)

    velocity_field = MLPVelocityField(2, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation)
    velocity_field.load_state_dict(torch.load(model_dir))

    particle_trajectories = torch.load(particle_trajectories_dir)
    particle_trajectories_prior = torch.load(particle_trajectories_prior_dir)
    particle_trajectories = particle_trajectories.detach().cpu().numpy()
    particle_trajectories_prior = particle_trajectories_prior.detach().cpu().numpy()
    if seed:
        g = torch.Generator().manual_seed(seed)  # set your seed here
        selected_idx = torch.randint(0, particle_trajectories.shape[0], (num_selection,), generator=g)
    else:
        selected_idx = torch.randint(0, particle_trajectories.shape[0], (num_selection,))

    num_timesteps = particle_trajectories.shape[1]
    timesteps = torch.linspace(0, 1, num_timesteps)

    particle_trajectories_selected = particle_trajectories[selected_idx, :, :]
    particle_trajectories_prior_selected = particle_trajectories_prior[selected_idx, :, :]

    particle_trajectories_selected_init_points = torch.from_numpy(particle_trajectories_selected[:, 0, :])
    particle_trajectories_ode_selected = odeint(velocity_field, 
                                                particle_trajectories_selected_init_points, 
                                                timesteps, 
                                                method=ode_solver)
    particle_trajectories_ode_selected = particle_trajectories_ode_selected.transpose(1,0).cpu().detach().numpy()


    print(f'Relative error: {100*np.linalg.norm(particle_trajectories_ode_selected - particle_trajectories_selected) / np.linalg.norm(particle_trajectories_selected):.4f}%')
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for i, traj in enumerate([particle_trajectories_prior_selected, 
                              particle_trajectories_selected, 
                              particle_trajectories_ode_selected]):
        
        for j in range(len(selected_idx)):
            color = plt.cm.viridis(j / len(selected_idx))
            ax[i].plot(traj[j, :, 0], traj[j, :, 1], '-', color="black")
            if j == 0:
                ax[i].plot(traj[j, 0, 0], traj[j, 0, 1], label=r'$X[t=0]$', marker='o', markersize=5, color='blue')
                ax[i].plot(traj[j, -1, 0], traj[j, -1, 1], label=r'$X[t=1]$', marker='o', markersize=5, color='red')
            else:
                ax[i].plot(traj[j, 0, 0], traj[j, 0, 1], marker='o', markersize=5, color='blue')
                ax[i].plot(traj[j, -1, 0], traj[j, -1, 1], marker='o', markersize=5, color='red')
            # Also, give a small label on top of the starting point
            c = 0.15
            ax[i].text(traj[j, 0, 0], traj[j, 0, 1] + c, f'{j}', fontsize=16)

    p_samples = generate_checkerboard_2d(sample_size, img_dir)
    q_samples = torch.randn_like(p_samples)

    for i, a in enumerate(ax.flatten()):
        a.scatter(p_samples[:, 0], p_samples[:, 1], s=0.01, alpha=0.5, color='blue')
        a.scatter(q_samples[:, 0], q_samples[:, 1], s=0.01, alpha=0.5, color='red')
        #a.set_xlim(-6, 6)
        #a.set_ylim(-6, 6)
        fsize = 16
        a.legend(fontsize=fsize, markerscale=3, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=2)
        a.tick_params(axis='both', which='major', labelsize=fsize - 2)
        if i == 0:
            a.set_title("Initial Particle Trajectories", fontsize=fsize)
        elif i == 1:
            a.set_title("Optimized Particle Trajectories", fontsize=fsize)
        elif i == 2:
            a.set_title("Trajectories by solving ODE", fontsize=fsize)


def plot_particle_trajectories_toy_example(model_dir,
                                           config_dir,
                                           img_dir,
                                           particle_trajectories_dir,
                                           num_selection,
                                           ode_solver,
                                           sample_size,
                                           seed=None,
                                           saving=None):
    
    epoch = model_dir.split("/")[-2]
    epoch_num = epoch.split("_")[-1]
    config = load_data(config_dir)

    velocity_field = MLPVelocityField(
        2,
        1,  # time dim = 1
        config.velocity_field_hidden_dims,
        config.velocity_field_layer_type,
        config.velocity_field_activation,
    )
    velocity_field.load_state_dict(torch.load(model_dir, map_location=torch.device("cpu")))

    particle_trajectories = torch.load(particle_trajectories_dir, map_location=torch.device("cpu"))
    particle_trajectories = particle_trajectories.detach().cpu().numpy()
    if seed:
        g = torch.Generator().manual_seed(seed)
        selected_idx = torch.randint(
            0, particle_trajectories.shape[0], (num_selection,), generator=g
        )
    else:
        selected_idx = torch.randint(
            0, particle_trajectories.shape[0], (num_selection,)
        )

    num_timesteps = particle_trajectories.shape[1]
    timesteps = torch.linspace(0, 1, num_timesteps)

    particle_trajectories_selected = particle_trajectories[selected_idx, :, :]
    particle_trajectories_selected_init_points = torch.from_numpy(
        particle_trajectories_selected[:, 0, :]
    )
    particle_trajectories_ode_selected = odeint(
        velocity_field,
        particle_trajectories_selected_init_points,
        timesteps,
        method=ode_solver,
    )
    particle_trajectories_ode_selected = (
        particle_trajectories_ode_selected.transpose(1, 0)
        .cpu()
        .detach()
        .numpy()
    )

    print(
        f"Relative error: "
        f"{100*np.linalg.norm(particle_trajectories_ode_selected - particle_trajectories_selected) / np.linalg.norm(particle_trajectories_selected):.4f}%"
    )

    # --- PLOT FRAMING STUFF STARTS HERE ---
    fsize = 13
    xlim = (-4.5, 4.5)
    ylim = (-4.5, 4.5)

    fig, ax = plt.subplots(
        1, 2, figsize=(8, 6), sharex=True, sharey=True
    )  # share axes for consistent framing
    # --- PLOT FRAMING STUFF ENDS HERE ---

    for i, traj in enumerate(
        [particle_trajectories_selected, particle_trajectories_ode_selected]
    ):
        for j in range(len(selected_idx)):
            ax[i].plot(traj[j, :, 0], traj[j, :, 1], "-", color="black", lw=0.8)
            # start (red) / end (blue)
            ax[i].plot(
                traj[j, 0, 0],
                traj[j, 0, 1],
                marker="o",
                markersize=5,
                color="red",
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
            ax[i].plot(
                traj[j, -1, 0],
                traj[j, -1, 1],
                marker="o",
                markersize=5,
                color="blue",
                markeredgecolor="white",
                markeredgewidth=0.8,
            )

    p_samples = generate_checkerboard_2d(sample_size, img_dir)
    q_samples = torch.randn_like(p_samples)

    for i, a in enumerate(ax.flatten()):
        # lighter, smaller background clouds
        a.scatter(
            p_samples[:, 0],
            p_samples[:, 1],
            s=1,
            alpha=0.08,
            color="tab:blue",
            rasterized=True,
        )
        a.scatter(
            q_samples[:, 0],
            q_samples[:, 1],
            s=1,
            alpha=0.10,
            color="tab:red",
            rasterized=True,
        )

        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.set_aspect("equal", "box")  # keep geometry undistorted

        # clean spines & ticks
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(axis="both", which="major", labelsize=fsize - 2)
        a.grid(alpha=0.2, linewidth=0.5)

        if i == 0:
            a.set_title("trajectories before flow matching", fontsize=fsize, pad=8)
        elif i == 1:
            a.set_title(
                "trajectories after flow matching", fontsize=fsize, pad=8
            )

    fig.suptitle(f"training epoch: {epoch_num}", fontsize=fsize + 2, y=1.02)
    fig.tight_layout()

    if saving:
        saving_dir = os.path.join(
            saving, f"particles_ode_trajectories_{epoch}.pdf"
        )
        fig.savefig(saving_dir, dpi=300, bbox_inches="tight")


def plot_2d_ode_trajectories(velocity_field, p, q, num_timesteps, num_plots, num_grid, ode_solver, odeint_batch_size=None, 
                             kde_bandwidth=0.01, device="cpu", saving=None):

    timesteps = torch.linspace(0, 1, num_timesteps).to(device)
    timesteps_reverse = torch.flip(timesteps, dims=[0])

    with torch.no_grad():  # Don't track gradients for ODE solving
        if odeint_batch_size:
            X_bar_fwd = batched_odeint(velocity_field, 
                                        p, 
                                        timesteps,
                                        odeint_batch_size,
                                        ode_solver=ode_solver,
                                        device=device)
            X_bar_bwd = batched_odeint(velocity_field, 
                                        q, 
                                        timesteps_reverse,
                                        odeint_batch_size,
                                        ode_solver=ode_solver,
                                        device=device)
        else:
            X_bar_fwd = odeint(velocity_field, p, timesteps, method=ode_solver) 
            X_bar_bwd = odeint(velocity_field, q, timesteps_reverse, method=ode_solver) 
        # (len(timesteps), training_size, dim)

    X_bar_fwd = X_bar_fwd.cpu().detach().numpy()
    X_bar_bwd = X_bar_bwd.cpu().detach().numpy()

    max_plot = min(num_plots, num_timesteps)
    idx = torch.linspace(0, num_timesteps - 1, max_plot).long()

    # Forward plots
    fig_forward, ax_forward = plt.subplots(1, max_plot, figsize=(max_plot * 3, 3), sharex=True, sharey=True)
    for i in range(max_plot):
        kde_forward = gaussian_kde(X_bar_fwd[idx[i], :, :].T, bw_method=kde_bandwidth)
        x_grid, y_grid = np.meshgrid(np.linspace(-4, 4, num_grid), np.linspace(-4, 4, num_grid))
        z_forward = kde_forward(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
        ax_forward[i].imshow(z_forward, extent=(-4, 4, -4, 4), origin='lower', cmap='viridis', alpha=0.5)
        ax_forward[i].set_title(f'$t={timesteps[idx[i]]:.2f}$', fontsize=18)

    fig_forward.tight_layout()
    plt.show()

    if saving:
        fig_forward.savefig(os.path.join(saving, "ode_trajectories_forward.pdf"),
                            dpi=300, bbox_inches="tight")
        
    plt.close(fig_forward)

    # Backward plots
    fig_backward, ax_backward = plt.subplots(1, max_plot, figsize=(max_plot * 3, 3), sharex=True, sharey=True)
    for i in range(max_plot):
        kde_backward = gaussian_kde(X_bar_bwd[idx[i], :, :].T, bw_method=kde_bandwidth)
        z_backward = kde_backward(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
        ax_backward[i].imshow(z_backward, extent=(-4, 4, -4, 4), origin='lower', cmap='viridis', alpha=0.5)
        ax_backward[i].set_title(f'$t={timesteps[num_timesteps-1-idx[i]]:.2f}$', fontsize=18)

    fig_backward.tight_layout()
    plt.show()

    if saving:
        fig_backward.savefig(os.path.join(saving, "ode_trajectories_backward.pdf"),
                            dpi=300, bbox_inches="tight")
    
    plt.close(fig_backward)
    

def display_multiple_images(images, rows, cols, figsize=1, titles=None, fontsize=14, saving=None):

    K = len(images)  # Number of available images
    fig, ax = plt.subplots(rows, cols, figsize=(int(figsize * cols), int(figsize * rows)), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            ax_now = ax[i, j]
            index = i * cols + j
            if index < K:
                grid_img_gen = torchvision.utils.make_grid(images[index], nrow=1)
                grid_img_gen = grid_img_gen.permute(1, 2, 0).detach().cpu().numpy()
                ax_now.imshow(grid_img_gen)
                ax_now.axis('off')
                if titles is not None and i == 0:
                    ax_now.set_title(titles[index], fontsize = fontsize)
            else:
                ax_now.axis('off')
    fig.subplots_adjust(hspace=0, wspace=0)

    if saving:
        fig.savefig(saving, dpi=200, bbox_inches="tight")


def visualize_decoded_samples_trajectories(decoded_samples, num_timestep_plots, saving):

    num_timesteps = decoded_samples.shape[0]
    num_samples = decoded_samples.shape[1]
    timestep_idx = torch.linspace(0, num_timesteps-1, num_timestep_plots).long()

    selected_decoded_samples = decoded_samples[timestep_idx,:,:,:,:]
    selected_decoded_samples = selected_decoded_samples.permute(1,0,2,3,4)
    selected_decoded_samples = selected_decoded_samples.flatten(0,1)
    
    image_list = []
    for i in range(selected_decoded_samples.shape[0]):
        image_list.append(selected_decoded_samples[i])

    if saving:
        display_multiple_images(image_list, 
                                rows=num_samples, 
                                cols=num_timestep_plots,
                                saving=saving)
    else:
        display_multiple_images(image_list, 
                                rows=num_samples, 
                                cols=num_timestep_plots,
                                saving=None)
        

def visualize_plain_samples(plain_samples, saving):

    image_list = []
    for i in range(plain_samples.shape[0]):
        image_list.append(plain_samples[i])

    if saving:
        display_multiple_images(image_list, 
                                rows=1, 
                                cols=len(image_list),
                                saving=saving)
    else:
        display_multiple_images(image_list, 
                                rows=1, 
                                cols=len(image_list),
                                saving=None)
        

def visualize_test_samples(test_samples, generated_sample_num, generated_sample_replication, saving):

    image_list = []
    for i in range(test_samples.shape[0]):
        image_list.append(test_samples[i])

    if saving:
        display_multiple_images(image_list, 
                                rows=generated_sample_replication, 
                                cols=generated_sample_num,
                                saving=saving)
    else:
        display_multiple_images(image_list, 
                                rows=generated_sample_replication, 
                                cols=generated_sample_num,
                                saving=None)
        

def plot_2d_gaussian_samples(samples_result, saving):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Subplot 1 ---
    axes[0].scatter(samples_result[0][:10000,0].detach().cpu(), samples_result[0][:10000,1].detach().cpu(), 
                    s=10, alpha=0.4, rasterized=True)
    axes[0].set_title("p")

    # --- Subplot 2 ---
    axes[1].scatter(samples_result[1][:10000,0].detach().cpu(), samples_result[1][:10000,1].detach().cpu(), 
                    s=10, alpha=0.4, rasterized=True)
    axes[1].set_title("q by benchmark")

    # --- Subplot 3 ---
    axes[2].scatter(samples_result[2][:10000,0].detach().cpu(), samples_result[2][:10000,1].detach().cpu(), 
                    s=10, alpha=0.4, rasterized=True)
    axes[2].set_title("q mapped")

    plt.tight_layout()
    plt.savefig(saving)


def plot_random_otceleba_images(benchmark, velocity_field, num_timesteps, ode_solver, 
                                sample_num, saving_dir, device):
    """
    test plot for random samples after the initialization
    """

    input_dim = (3,64,64)
    p_batch = benchmark.input_sampler.sample(sample_num)
    timesteps = torch.linspace(0, 1, num_timesteps)

    with torch.no_grad():
        p_trajectory_batch = odeint(velocity_field, 
                                    p_batch.reshape(-1, *input_dim).to(device), 
                                    timesteps.to(device), 
                                    method=ode_solver) 
        
    p1_batch = p_trajectory_batch[-1,:,:,:,:] # (sample_size, 3, 64, 64)
    p_batch.requires_grad_(True)
    p_mapped = benchmark.map_fwd(p_batch, nograd=True) #dims?
    p_mapped = p_mapped.reshape(-1, *input_dim)

    if p1_batch.shape != p_mapped.shape:
        raise ValueError("the shape of input solutions must be matched")
    
    # (2*sample_num, 64, 64, 3)
    imgs_vec = np.concatenate([p1_batch.cpu().detach().permute(0,2,3,1).mul(0.5).add(0.5).cpu().numpy().clip(0, 1), 
                               p_mapped.cpu().detach().permute(0,2,3,1).mul(0.5).add(0.5).cpu().numpy().clip(0, 1)])
    
    fig, axes = plt.subplots(2, sample_num, figsize=(12, 4), dpi=100)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs_vec[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
    
    axes[0, 0].set_ylabel('velocity field', fontsize=12)
    axes[1, 0].set_ylabel('OT', fontsize=12)
    fig.tight_layout()

    if saving_dir:
        plt.savefig(saving_dir)

    plt.close(fig) 