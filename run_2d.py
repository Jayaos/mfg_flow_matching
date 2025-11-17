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

    timesteps = torch.linspace(0, 1, config.num_timesteps)
    timestep_size = 1/(config.num_timesteps-1)
    input_dim = 2

    # initialize model
    classifier = MLPClassifier(input_dim, config.classifier_hidden_dims, activation=config.classifier_activation).to(device)
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.velocity_field_learning_rate)
    initial_classifier_loss_record = []
    loss_record = dict()
    initial_seed = config.seed

    mfgflow_training_pbar = tqdm(total=config.epochs, 
                                 desc="MFG-Flow Training Epochs")

    for i in range(config.epochs):

        epoch_saving_dir = os.path.join(config.saving_dir, "epoch_{}".format(i+1))
        os.makedirs(epoch_saving_dir, exist_ok=True)

        # generate data for training
        p_training, _ = generate_toy_data(p_dataset_config, initial_seed)
        q_training, _ = generate_toy_data(q_dataset_config, initial_seed)
        initial_seed += 1

        if i == 0:

            if config.velocity_field_initialization:
                print("Initialize particle trajectories with flow matching on linear interpolant at the first epoch")

                vf_init_optim = torch.optim.Adam(velocity_field.parameters(), 
                                                 lr=config.particle_optimization_learning_rate)
                vf_init_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_training, q_training), 
                                                                   batch_size=config.velocity_field_training_batch_size, 
                                                                   shuffle=True))
                vf_init_pbar = tqdm(total=config.velocity_field_initialization_training_step, 
                                    desc="Velocity Fields Initialization")
                
                if config.velocity_field_initialization in ["interflow", "stochastic-interpolant"]:
                    flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
                elif config.velocity_field_initialization in ["conditional-flow-matching"]:
                    flow_matcher = ConditionalFlowMatcher(sigma=0.1)
                elif config.velocity_field_initialization in ["linear-flow-matching"]:
                    flow_matcher = LinearFlowMatcher()

                for p_batch, q_batch in vf_init_dataloader:
                    if vf_init_dataloader.step <= config.velocity_field_initialization_training_step:
                        
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
                    if config.odeint_batch_size:
                        X_bar = batched_odeint(velocity_field, 
                                               p_training, 
                                               timesteps,
                                               config.odeint_batch_size,
                                               ode_solver=config.ode_solver,
                                               device=device)
                    else:
                        X_bar = odeint(velocity_field, p_training, timesteps, method=config.ode_solver) 
                    # (len(timesteps), training_size, dim)

            else:
                print("Initialize particle trajectories with linear interpolant at the first epoch")
                # X_bar: (len(timesteps), data_size, dim)
                X_bar = initialize_linear_interpolant(p_training, q_training, timesteps)
                # initial Xbar is just linear interpolant of p and q based on time steps

            classifier_optimization_pbar = tqdm(total=config.initial_classifier_training_step, 
                                                desc="Initial Classifier Training Steps")
            p_1_training = X_bar[-1, :, :].detach() # (data_size, dim), endpoint of linear interpolant
            classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training), 
                                                                  batch_size=config.classifier_training_batch_size, 
                                                                  shuffle=True))

            for x_p1, x_q in classifier_dataloader:
                x_p1 = x_p1.to(device)
                x_q = x_q.to(device)
                if classifier_dataloader.step <= config.initial_classifier_training_step:
                    classifier_loss = classifier_loss_fn(classifier, x_p1, x_q)
                    classifier_optim.zero_grad()
                    classifier_loss["loss"].backward()
                    classifier_optim.step()
                    initial_classifier_loss_record.append(classifier_loss["loss"].item())
                    classifier_optimization_pbar.update(1)
                else:
                    # config.initial_classifier_training_step reached break the for loop
                    break

        else:
            # if its not the first epoch, Xbar is trajectory obtained by solving ODE
            ## TODO: potentially using ODESolver class is ideal
            # ode_solver = ODESolver(velocity_field)
            # X_bar = ode_solver.sample()
            print("Obtaining particle trajectories by solving ODE using {}".format(config.ode_solver))

            with torch.no_grad():  # Don't track gradients for ODE solving
                if config.odeint_batch_size:
                    X_bar = batched_odeint(velocity_field, 
                                           p_training, 
                                           timesteps,
                                           config.odeint_batch_size,
                                           ode_solver=config.ode_solver,
                                           device=device)
                else:
                    X_bar = odeint(velocity_field, p_training, timesteps, method=config.ode_solver) 
                # (len(timesteps), training_size, dim)

        # Particle optimization
        ## before particle optimization, generate X_bar through initialization or solving ODE first
        ## and then batchify them, since batchfiy > solving ODE is more time consuming

        # Prepare trajectory variables
        particle_0 = X_bar[0].clone().detach().unsqueeze(0).to(device) # (1, data_size, dim)
        particle_trajectory = X_bar[1:].clone().detach().to(device) # (num_timesteps - 1, data_size, dim)
        particle_trajectory.requires_grad_(True)
        particle_optim = torch.optim.Adam([particle_trajectory], lr=config.particle_optimization_learning_rate)

        if i == 0:
            particle_optimization_epoch = config.initial_particle_optimization_epoch
        else:
            particle_optimization_epoch = config.particle_optimization_epoch

        kinetic_loss_record = []
        classifier_loss_record = []
        particle_optimization_loss_record = []

        if config.particle_optimization_batch_size:
            # batch optimization of particle trajectory

            particle_dataloader = TrajectoryDataLoader(particle_0, 
                                                       particle_trajectory,
                                                       batch_size=config.particle_optimization_batch_size, 
                                                       shuffle=True)
            particle_dataloader_size = len(particle_dataloader)
            particle_optimization_pbar = tqdm(total=particle_optimization_epoch, 
                                            desc="Particle Opimization Epochs")
            intermediate_classifier_loss_record = []
                
            for e in tqdm(range(particle_optimization_epoch)):
                
                kinetic_loss_sum = 0.
                classifier_loss_sum = 0.
                particle_optimization_loss_sum = 0.

                for particle_0_batch, particle_trajectory_batch in tqdm(particle_dataloader, 
                                                                    total=len(particle_dataloader), 
                                                                    desc=f"Epoch {e+1}/{particle_optimization_epoch}"):

                    particle_optimization_loss = particle_optimization_loss_fn(particle_0_batch, 
                                                                            particle_trajectory_batch,
                                                                            classifier,
                                                                            kinetic_loss_weight=config.kinetic_loss_weight)
                    particle_optim.zero_grad()
                    particle_optimization_loss["loss"].backward()
                    particle_optim.step()

                    kinetic_loss_sum += particle_optimization_loss["kinetic_loss"].detach().cpu().item()
                    classifier_loss_sum += particle_optimization_loss["classifier_loss"].detach().cpu().item()
                    particle_optimization_loss_sum += particle_optimization_loss["loss"].detach().cpu().item()

                kinetic_loss_record.append(kinetic_loss_sum / particle_dataloader_size)
                classifier_loss_record.append(classifier_loss_sum / particle_dataloader_size)
                particle_optimization_loss_record.append(particle_optimization_loss_sum / particle_dataloader_size)

                # Update classifier every freq_update epochs
                if (e + 1) % config.classifier_intermediate_training_frequency == 0:

                    print("Intermediate Classifier Training at Particle Optimization Epoch {}...".format(e+1))

                    p_1_training = particle_trajectory[-1, :, :].clone().detach() # (data_size, dim)
                    classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training), 
                                                                        batch_size=config.classifier_training_batch_size, 
                                                                        shuffle=True))
                    for x_p, x_q in classifier_dataloader:
                        x_p = x_p.to(device)
                        x_q = x_q.to(device)
                        if classifier_dataloader.step <= config.intermediate_classifier_training_step:
                            classifier_loss = classifier_loss_fn(classifier, x_p, x_q)
                            classifier_optim.zero_grad()
                            classifier_loss["loss"].backward()
                            classifier_optim.step()
                            intermediate_classifier_loss_record.append(classifier_loss["loss"].item())
                        else:
                            break

                particle_optimization_pbar.update(1)

        else:
            # entire trajectory optimization in one epoch

            particle_optimization_pbar = tqdm(total=particle_optimization_epoch, 
                                              desc="Particle Opimization Epochs")
            intermediate_classifier_loss_record = []

            for e in tqdm(range(particle_optimization_epoch)):
                
                kinetic_loss_sum = 0.
                classifier_loss_sum = 0.
                particle_optimization_loss_sum = 0.
                particle_optimization_loss = particle_optimization_loss_fn(particle_0,  
                                                                           particle_trajectory, 
                                                                           classifier,
                                                                           kinetic_loss_weight=config.kinetic_loss_weight)
                particle_optim.zero_grad()
                particle_optimization_loss["loss"].backward()
                particle_optim.step()

                kinetic_loss_record.append(particle_optimization_loss["kinetic_loss"].detach().cpu().item())
                classifier_loss_record.append(particle_optimization_loss["classifier_loss"].detach().cpu().item())
                particle_optimization_loss_record.append(particle_optimization_loss["loss"].detach().cpu().item())

                # Update classifier every freq_update epochs
                if (e + 1) % config.classifier_intermediate_training_frequency == 0:

                    print("Intermediate Classifier Training at Particle Optimization Epoch {}...".format(e+1))

                    p_1_training = particle_trajectory[-1, :, :].clone().detach() # (data_size, dim)
                    classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training), 
                                                                        batch_size=config.classifier_training_batch_size, 
                                                                        shuffle=True))
                    for x_p, x_q in classifier_dataloader:
                        x_p = x_p.to(device)
                        x_q = x_q.to(device)
                        if classifier_dataloader.step <= config.intermediate_classifier_training_step:
                            classifier_loss = classifier_loss_fn(classifier, x_p, x_q)
                            classifier_optim.zero_grad()
                            classifier_loss["loss"].backward()
                            classifier_optim.step()
                            intermediate_classifier_loss_record.append(classifier_loss["loss"].item())
                        else:
                            break

                particle_optimization_pbar.update(1)

        # this is updated particle trajectory
        X_bar = torch.cat([particle_0.detach().cpu(), particle_trajectory.detach().cpu()], dim=0) # (len(timesteps), training_size, dim)
        X_bar = X_bar.transpose(1,0) # (training_size, len(timesteps), channel, h, w)

        if i == 0:
            # at the first epoch, train velocity field more as warm-up
            velocity_field_training_step = config.initial_velocity_field_training_step
        else:
            velocity_field_training_step = config.velocity_field_training_step

        vf_dataloader = DataLoaderIterator(DataLoader(TensorDataset(X_bar), 
                                            batch_size=config.velocity_field_training_batch_size, 
                                            shuffle=True))
        
        vf_loss_record = []
        vf_pbar = tqdm(total=velocity_field_training_step,
                       desc="Velocity Field Training Steps")
        
        for x_traj_batch in vf_dataloader:
            
            # x_traj_batch is a tuple so x_traj_batch[0] is needed
            if vf_dataloader.step <= velocity_field_training_step:
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
        
        # generate test data
        _ , p_test = generate_toy_data(p_dataset_config, config.seed) # use same seed across epoch
        _ , q_test = generate_toy_data(q_dataset_config, config.seed) # use same seed across epoch

        # trajectory plot
        plot_2d_ode_trajectories(velocity_field, 
                                 p_test, q_test, 
                                 config.num_timesteps, 
                                 8, 200, "rk4", 2048, 
                                 device=device, 
                                 saving=epoch_saving_dir)

        # test path energy
        path_energy = compute_path_energy(velocity_field, 
                                          p_test, 
                                          config.num_timesteps,
                                          config.odeint_batch_size,
                                          config.ode_solver,
                                          device=device)
        print("path energy at epoch {}: {}".format(i+1, path_energy))
        mfgflow_training_pbar.update(1)

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
        torch.save(classifier.state_dict(), os.path.join(epoch_saving_dir, 'classifier_e{}.pt'.format(i+1)))
        torch.save(velocity_field.state_dict(), os.path.join(epoch_saving_dir, 'velocity_field_e{}.pt'.format(i+1)))
    
