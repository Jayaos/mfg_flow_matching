from config import MFGFlowToyExampleConfig
from model import MLPClassifier, MLPVelocityField
from model import classifier_loss_fn, particle_optimization_loss_fn, vf_loss_fn
from utils import DataLoaderIterator, TrajectoryDataLoader
from utils import initialize_linear_interpolant, batched_odeint
from utils import load_data, save_data, generate_toy_data, compute_path_energy
from utils.plotting import plot_2d_ode_trajectories
from baselines.torchcfm.conditional_flow_matching import VariancePreservingConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import LinearFlowMatcher
from torchdiffeq import odeint
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os


def run_mfg_flow_toy_example(config: MFGFlowToyExampleConfig, p_dataset_config, q_dataset_config, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "p_dataset_config.pkl", p_dataset_config)
    save_data(config.saving_dir + "q_dataset_config.pkl", q_dataset_config)
    save_data(config.saving_dir + "mfg_flow_config.pkl", config)

    p_training, p_test = generate_toy_data(p_dataset_config, config.seed)
    q_training, q_test = generate_toy_data(q_dataset_config, config.seed)

    outer_loop_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_training, q_training), 
                                                          batch_size=config.outer_batch, 
                                                          shuffle=True, 
                                                          drop_last=True))
    
    timesteps = torch.linspace(0, 1, config.ode_timesteps)
    timestep_size = 1/(config.ode_timesteps-1)
    input_dim = 2

    # initialize model
    classifier = MLPClassifier(input_dim, config.classifier_hidden_dims, activation=config.classifier_activation).to(device)
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.vf_hidden_dims, 
                                      config.vf_layer_type,
                                      config.vf_activation).to(device)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.vf_learning_rate)
    initial_classifier_loss_record = []
    loss_record = dict()

    outer_loop_pbar = tqdm(total=config.outer_loop, 
                           desc="MFG-Flow Training Outer Loop")

    for i in range(config.outer_loop):

        loop_saving_dir = os.path.join(config.saving_dir, "loop_{}".format(i+1))
        os.makedirs(loop_saving_dir, exist_ok=True)

        p_training, q_training = next(outer_loop_dataloader)

        if i == 0:

            if config.vf_initialization:
                print("Initialize particle trajectories with flow matching on linear interpolant at the first loop")

                vf_init_optim = torch.optim.Adam(velocity_field.parameters(), 
                                                 lr=config.particle_learning_rate)
                vf_init_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_training, q_training), 
                                                                   batch_size=config.vf_minibatch, 
                                                                   shuffle=True))
                vf_init_pbar = tqdm(total=config.vf_initial_steps, 
                                    desc="Velocity Fields Initialization")
                
                if config.vf_initialization in ["interflow", "stochastic-interpolant"]:
                    flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
                elif config.vf_initialization in ["conditional-flow-matching"]:
                    flow_matcher = ConditionalFlowMatcher(sigma=0.1)
                elif config.vf_initialization in ["linear-flow-matching"]:
                    flow_matcher = LinearFlowMatcher()

                for p_batch, q_batch in vf_init_dataloader:
                    if vf_init_dataloader.step <= config.vf_initial_steps:
                        
                        t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch,
                                                                                                        q_batch)
                        t_batch = t_batch.to(device)
                        xt_batch = xt_batch.to(device)
                        ut_batch = ut_batch.to(device)
                        vt_batch = velocity_field(t_batch, xt_batch)
                        loss = torch.mean((vt_batch - ut_batch)**2)
                        vf_init_optim.zero_grad()
                        loss.backward()
                        vf_init_optim.step()
                        vf_init_pbar.update(1)
                    else:
                        break

                with torch.no_grad():  # Don't track gradients for ODE solving
                    if config.odeint_minibatch:
                        X_bar = batched_odeint(velocity_field, 
                                               p_training, 
                                               timesteps,
                                               config.odeint_minibatch,
                                               ode_solver=config.ode_solver,
                                               device=device)
                    else:
                        X_bar = odeint(velocity_field, p_training, timesteps, method=config.ode_solver) 
                    # (len(timesteps), training_size, dim)

            else:
                print("Initialize particle trajectories with linear interpolant")
                # X_bar: (len(timesteps), data_size, dim)
                X_bar = initialize_linear_interpolant(p_training, timesteps)
                # initial Xbar is just linear interpolant of p and q based on time steps

            classifier_optimization_pbar = tqdm(total=config.classifier_initial_steps, 
                                                desc="Initial Classifier Training Steps")
            p_1_training = X_bar[-1, :, :].detach() # (data_size, dim), endpoint of linear interpolant
            classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training), 
                                                                  batch_size=config.classifier_minibatch, 
                                                                  shuffle=True))

            for x_p1, x_q in classifier_dataloader:
                x_p1 = x_p1.to(device)
                x_q = x_q.to(device)
                if classifier_dataloader.step <= config.classifier_initial_steps:
                    classifier_loss = classifier_loss_fn(classifier, x_p1, x_q)
                    classifier_optim.zero_grad()
                    classifier_loss["loss"].backward()
                    classifier_optim.step()
                    initial_classifier_loss_record.append(classifier_loss["loss"].item())
                    classifier_optimization_pbar.update(1)
                else:
                    # config.classifier_initial_steps reached break the for loop
                    break

        else:
            # if its not the first epoch, Xbar is trajectory obtained by solving ODE
            print("Obtaining particle trajectories by solving ODE")

            with torch.no_grad():  # Don't track gradients for ODE solving
                if config.odeint_minibatch:
                    X_bar = batched_odeint(velocity_field, 
                                           p_training, 
                                           timesteps,
                                           config.odeint_minibatch,
                                           ode_solver=config.ode_solver,
                                           device=device)
                else:
                    X_bar = odeint(velocity_field, p_training, timesteps, method=config.ode_solver) 
                # (len(timesteps), training_size, dim)

        # Particle optimization
        ## before particle optimization, generate X_bar through initialization or solving ODE first
        ## and then batchify them, since batchfiy > solving ODE is more time consuming

        torch.save(X_bar.transpose(1,0), os.path.join(loop_saving_dir, 
                                                      'pre_optimized_particles_trajectories_l{}.pt'.format(i+1)))

        # Prepare trajectory variables
        particle_0 = X_bar[0].clone().detach().unsqueeze(0).to(device) # (1, data_size, dim)
        particle_trajectory = X_bar[1:].clone().detach().to(device) # (num_timesteps - 1, data_size, dim)
        particle_trajectory.requires_grad_(True)
        particle_optim = torch.optim.Adam([particle_trajectory], lr=config.particle_learning_rate)

        kinetic_loss_record = []
        classifier_loss_record = []
        particle_optimization_loss_record = []
        intermediate_classifier_loss_record = []
        particle_dataloader = DataLoaderIterator(TrajectoryDataLoader(particle_0, 
                                                                      particle_trajectory,
                                                                      batch_size=config.particle_minibatch, 
                                                                      shuffle=True))
            
        for particle_0_batch, particle_trajectory_batch in particle_dataloader:

            if particle_dataloader.step <= config.particle_steps:
                
                particle_optimization_loss = particle_optimization_loss_fn(particle_0_batch, 
                                                                            particle_trajectory_batch,
                                                                            classifier,
                                                                            kinetic_loss_weight=config.kinetic_loss_weight)
                particle_optim.zero_grad()
                particle_optimization_loss["loss"].backward()
                particle_optim.step()

                kinetic_loss_record.append(particle_optimization_loss["kinetic_loss"].detach().cpu().item())
                classifier_loss_record.append(particle_optimization_loss["classifier_loss"].detach().cpu().item())
                particle_optimization_loss_record.append(particle_optimization_loss["loss"].detach().cpu().item())

                if particle_dataloader.step % config.cost_update_frequency == 0:

                    p_1_training = particle_trajectory[-1, :, :].clone().detach() # (data_size, dim)
                    classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training), 
                                                                        batch_size=config.classifier_minibatch, 
                                                                        shuffle=True))
                    for x_p, x_q in classifier_dataloader:
                            x_p = x_p.to(device)
                            x_q = x_q.to(device)
                            if classifier_dataloader.step <= config.classifier_retrain_steps:
                                classifier_loss = classifier_loss_fn(classifier, x_p, x_q)
                                classifier_optim.zero_grad()
                                classifier_loss["loss"].backward()
                                classifier_optim.step()
                                intermediate_classifier_loss_record.append(classifier_loss["loss"].item())
                            else:
                                break

            else:
                break

        # this is updated particle trajectory
        X_bar = torch.cat([particle_0.detach().cpu(), particle_trajectory.detach().cpu()], dim=0) # (len(timesteps), training_size, dim)
        X_bar = X_bar.transpose(1,0) # (training_size, len(timesteps), channel, h, w)

        vf_dataloader = DataLoaderIterator(DataLoader(TensorDataset(X_bar), 
                                            batch_size=config.vf_minibatch, 
                                            shuffle=True))
        
        vf_loss_record = []
        vf_pbar = tqdm(total=config.vf_steps,
                       desc="Velocity Field Training Steps")
        
        for x_traj_batch in vf_dataloader:
            
            # x_traj_batch is a tuple so x_traj_batch[0] is needed
            if vf_dataloader.step <= config.vf_steps:
                # (batch_size, len(timesteps), dim)
                vf_loss = vf_loss_fn(velocity_field, 
                                     x_traj_batch[0].to(device), 
                                     timesteps.to(device), 
                                     timestep_size)
                vf_optim.zero_grad()
                vf_loss["loss"].backward()
                vf_optim.step()
                vf_loss_record.append(vf_loss["loss"].detach().cpu().item())
                vf_pbar.update(1)
            else:
                break

        # trajectory plot
        plot_2d_ode_trajectories(velocity_field, 
                                 p_test, q_test, 
                                 config.ode_timesteps, 
                                 8, 200, "rk4", 2048, 
                                 device=device, 
                                 saving=loop_saving_dir)

        # test path energy
        path_energy = compute_path_energy(velocity_field, 
                                          p_test, 
                                          config.ode_timesteps,
                                          config.odeint_minibatch,
                                          config.ode_solver,
                                          device=device)
        print("path energy at epoch {}: {}".format(i+1, path_energy))
        outer_loop_pbar.update(1)

        if i == 0:
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "initial_classifier_loss_record" : initial_classifier_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "intermediate_classifier_loss_record" : intermediate_classifier_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record,
                              "path_energy" : path_energy}
        else:                
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "intermediate_classifier_loss_record" : intermediate_classifier_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record,
                              "path_energy" : path_energy}
            
        save_data(config.saving_dir + "loss_record.pkl", loss_record)
        torch.save(X_bar, os.path.join(loop_saving_dir, 'optimized_particles_trajectories_l{}.pt'.format(i+1)))
        torch.save(classifier.state_dict(), os.path.join(loop_saving_dir, 'classifier_l{}.pt'.format(i+1)))
        torch.save(velocity_field.state_dict(), os.path.join(loop_saving_dir, 'velocity_field_l{}.pt'.format(i+1)))
    
