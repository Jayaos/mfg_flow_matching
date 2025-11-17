from config import MFGFlowImageConfig
from model.classifier import UNetClassifier
from model.velocity_field import ConvVelocityField
from model import classifier_loss_fn, particle_optimization_loss_fn, image_vf_loss_fn
from utils import load_image_dataset, DataLoaderIterator, TrajectoryDataLoader
from utils import initialize_X_bar_image, batched_odeint
from utils import visualize_decoded_samples_trajectories
from utils import load_data, save_data, compute_fid, compute_fid_endpoint
from ldm.models.utils import autoencoder_decode
from ldm.util import instantiate_from_config
from baselines.torchcfm.conditional_flow_matching import VariancePreservingConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import LinearFlowMatcher
from omegaconf import OmegaConf
from torchdiffeq import odeint
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os


def run_mfg_flow_image(config: MFGFlowImageConfig, dataset_config, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "dataset_config.pkl", dataset_config)
    save_data(config.saving_dir + "mfg_flow_config.pkl", config)

    image_dataset = load_image_dataset(config, dataset_config)

    # load trained VAE
    print("loading pre-trained autoencoder...")
    vae_config = OmegaConf.load(config.vae_config_dir)
    vae_model = instantiate_from_config(vae_config.model)
    vae_model.load_state_dict(torch.load(config.vae_model_dir)["state_dict"], strict=False)

    # no need to load VAE here for training
    timesteps = torch.linspace(0, 1, config.num_timesteps)
    timestep_size = 1/(config.num_timesteps-1)
    input_dim = image_dataset["p_training"][0].shape

    # initialize model
    print("data dimension: {}".format(input_dim))
    print("data size of P : {}".format(image_dataset["p_training"].shape))
    print("data size of Q : {}".format(image_dataset["q_training"].shape))
    print("data size used for each training epoch: {}".format(int(len(image_dataset["p_training"])*
                                                                  config.epoch_training_ratio)))

    classifier = UNetClassifier(input_dim[0], 
                                config.classifier_channels,
                                config.classifier_use_bias).to(device)
    velocity_field = ConvVelocityField(input_dim[0], 
                                      config.velocity_field_encoding_dims,
                                      config.velocity_field_decoding_dims,
                                      config.velocity_field_kernel_sizes,
                                      config.velocity_field_strides).to(device)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.velocity_field_learning_rate)

    initial_classifier_loss_record = []
    loss_record = dict()

    mfgflow_training_pbar = tqdm(total=config.epochs, 
                                 desc="MFG-Flow Training Epochs")

    for i in range(config.epochs):

        epoch_saving_dir = os.path.join(config.saving_dir, "epoch_{}".format(i+1))
        os.makedirs(epoch_saving_dir, exist_ok=True)

        # p_training and q_training have the same size
        # select small portion of training size for the epoch training
        epoch_training_size = int(len(image_dataset["p_training"])*config.epoch_training_ratio)
        rand_idx = torch.randperm(len(image_dataset["p_training"]))[:epoch_training_size]
        p_training_epoch = image_dataset["p_training"][rand_idx]
        q_training_epoch = image_dataset["q_training"][rand_idx]

        if i == 0:

            if config.velocity_field_initialization:
                print("Initialize particle trajectories with flow matching on linear interpolant at the first epoch")

                vf_init_optim = torch.optim.Adam(velocity_field.parameters(), 
                                                 lr=config.particle_optimization_learning_rate)
                vf_init_dataloader = DataLoaderIterator(DataLoader(TensorDataset(image_dataset["p_training"], 
                                                                                 image_dataset["q_training"]), 
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
                                            p_training_epoch, 
                                            timesteps,
                                            config.odeint_batch_size,
                                            ode_solver=config.ode_solver,
                                            device=device)
                    else:
                        X_bar = odeint(velocity_field, p_training_epoch, timesteps, method=config.ode_solver) 
                    # (len(timesteps), training_size, dims, ...)

            else:
                print("Initialize particle trajectories with linear interpolant at the first epoch")
                # X_bar: (len(timesteps), data_size, channel, w, h)
                X_bar = initialize_X_bar_image(p_training_epoch, q_training_epoch, timesteps)
                # initial Xbar is just linear interpolant of p and q based on time steps

            # select 5 images for sanity check
            rand_idx_visualization = torch.randperm(len(p_training_epoch))[:5]
            # (num_timesteps, len(rand_idx_visualization), channel, w, h)
            X_bar_visualization_samples = X_bar[:,rand_idx_visualization,:,:,:].flatten(0,1)
            vae_model.eval()
            decoded_samples = autoencoder_decode(vae_model, 
                                                 X_bar_visualization_samples, 
                                                 rescale_factor=image_dataset["q_rescale_factor"],
                                                 batch_size=None,
                                                 device=device)
            decoded_samples = decoded_samples.view((len(timesteps), 
                                                    len(rand_idx_visualization),
                                                    image_dataset["image_dim"][2],
                                                    image_dataset["image_dim"][0],
                                                    image_dataset["image_dim"][1])).cpu().detach()
            
            saving_file_name = os.path.join(epoch_saving_dir, "5decoded_samples_trajectories.pdf")
            visualize_decoded_samples_trajectories(decoded_samples, 8, saving_file_name)

            classifier_optimization_pbar = tqdm(total=config.initial_classifier_training_step, 
                                                desc="Initial Classifier Training Steps")
            p_1_training = X_bar[-1,:,:,:,:].detach() # (data_size, dim), endpoint of linear interpolant
            classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training_epoch), 
                                                                  batch_size=config.classifier_training_batch_size, 
                                                                  shuffle=True))
            
            # initialization FID test here
            vf_test_fid = compute_fid_endpoint(vae_model, 
                                                image_dataset["q_rescale_factor"], 
                                                p_1_training,
                                                image_dataset["q_true_dataloader"], 
                                                config.odeint_batch_size, 
                                                device)
            print("test FID of the initialized velocity field: {}".format(vf_test_fid))

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
            # Xbar is trajectory obtained by solving ODE
            ## TODO: potentially using ODESolver class is ideal
            # ode_solver = ODESolver(velocity_field)
            # X_bar = ode_solver.sample()
            print("Obtaining particle trajectories by solving ODE using {}".format(config.ode_solver))

            with torch.no_grad():  # Don't track gradients for ODE solving
                if config.odeint_batch_size:
                    X_bar = batched_odeint(velocity_field, 
                                           p_training_epoch, 
                                           timesteps,
                                           config.odeint_batch_size,
                                           ode_solver=config.ode_solver,
                                           device=device)
                else:
                    X_bar = odeint(velocity_field, p_training_epoch, timesteps, method=config.ode_solver) 
                # (len(timesteps), training_size, dims, ...)
                         
            # select 5 images for sanity check
            rand_idx_visualization = torch.randperm(len(p_training_epoch))[:5]
            # (num_timesteps, len(rand_idx_visualization), channel, w, h)
            X_bar_visualization_samples = X_bar[:,rand_idx_visualization,:,:,:].flatten(0,1)
            decoded_samples = autoencoder_decode(vae_model, 
                                                 X_bar_visualization_samples, 
                                                 rescale_factor=image_dataset["q_rescale_factor"],
                                                 batch_size=None,
                                                 device=device)
            decoded_samples = decoded_samples.view((len(timesteps), 
                                                    len(rand_idx_visualization),
                                                    image_dataset["image_dim"][2],
                                                    image_dataset["image_dim"][0],
                                                    image_dataset["image_dim"][1])).cpu().detach()
            
            saving_file_name = os.path.join(epoch_saving_dir, "5decoded_samples_trajectories.pdf")
            visualize_decoded_samples_trajectories(decoded_samples, 8, saving_file_name)

        # Particle optimization
        ## before particle optimization, generate X_bar through initialization or solving ODE first
        ## and then batchify them, since batchfiy > solving ODE is more time consuming

        # Prepare trajectory variables
        particle_0 = X_bar[0].clone().detach().unsqueeze(0).to(device) # (1, data_size, channel, h, w)
        particle_trajectory = X_bar[1:].clone().detach().to(device) # (num_timesteps - 1, data_size, channel, h, w)
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
                    
                    #current_kinetic_loss_weight = config.kinetic_loss_weight + ((i+1) // 10) * config.kinetic_loss_weight
                    current_kinetic_loss_weight = config.kinetic_loss_weight

                    particle_optimization_loss = particle_optimization_loss_fn(particle_0_batch, 
                                                                            particle_trajectory_batch,
                                                                            classifier,
                                                                            kinetic_loss_weight=current_kinetic_loss_weight)
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
                current_classifier_intermediate_training_frequency = (1+((i+1) // 10)) * config.classifier_intermediate_training_frequency
                #current_classifier_intermediate_training_frequency = config.classifier_intermediate_training_frequency
                if (e + 1) % current_classifier_intermediate_training_frequency == 0:

                    print("Intermediate Classifier Training at Particle Optimization Epoch {}...".format(e+1))

                    p_1_training = particle_trajectory[-1, :, :, :, :].clone().detach() # (data_size, dims)
                    classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training_epoch), 
                                                                        batch_size=config.classifier_training_batch_size, 
                                                                        shuffle=True))
                    
                    #current_intermediate_classifier_training_step = (1+((i+1) // 10)) * config.intermediate_classifier_training_step 
                    current_intermediate_classifier_training_step = config.intermediate_classifier_training_step

                    for x_p, x_q in classifier_dataloader:
                        x_p = x_p.to(device)
                        x_q = x_q.to(device)
                        if classifier_dataloader.step <= current_intermediate_classifier_training_step:
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

                #current_kinetic_loss_weight = config.kinetic_loss_weight + ((i+1) // 10) * config.kinetic_loss_weight
                current_kinetic_loss_weight = config.kinetic_loss_weight
            
                kinetic_loss_sum = 0.
                classifier_loss_sum = 0.
                particle_optimization_loss_sum = 0.
                particle_optimization_loss = particle_optimization_loss_fn(particle_0,  
                                                                           particle_trajectory, 
                                                                           classifier,
                                                                           kinetic_loss_weight=current_kinetic_loss_weight)
                particle_optim.zero_grad()
                particle_optimization_loss["loss"].backward()
                particle_optim.step()

                kinetic_loss_record.append(particle_optimization_loss["kinetic_loss"].detach().cpu().item())
                classifier_loss_record.append(particle_optimization_loss["classifier_loss"].detach().cpu().item())
                particle_optimization_loss_record.append(particle_optimization_loss["loss"].detach().cpu().item())

                # Update classifier every freq_update epochs
                current_classifier_intermediate_training_frequency = (1+((i+1) // 10)) * config.classifier_intermediate_training_frequency
                #current_classifier_intermediate_training_frequency = config.classifier_intermediate_training_frequency
                if (e + 1) % current_classifier_intermediate_training_frequency == 0:

                    print("Intermediate Classifier Training at Particle Optimization Epoch {}...".format(e+1))

                    p_1_training = particle_trajectory[-1, :, :, :, :].clone().detach() # (data_size, dim)
                    classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training_epoch), 
                                                                        batch_size=config.classifier_training_batch_size, 
                                                                        shuffle=True))
                    
                    #current_intermediate_classifier_training_step = (1+((i+1) // 10)) * config.intermediate_classifier_training_step 
                    current_intermediate_classifier_training_step = config.intermediate_classifier_training_step

                    for x_p, x_q in classifier_dataloader:
                        x_p = x_p.to(device)
                        x_q = x_q.to(device)
                        if classifier_dataloader.step <= current_intermediate_classifier_training_step:
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
                # (batch_size, len(timesteps), channel, w, h)
                vf_loss = image_vf_loss_fn(velocity_field, 
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

        mfgflow_training_pbar.update(1)

        # compute FID
        particle_test_fid, vf_test_fid, particle_rec_test_fid, \
        vf_rec_test_fid, vf_particle_fid = compute_fid(velocity_field, 
                                                        vae_model,
                                                        image_dataset["q_rescale_factor"],  
                                                        X_bar,
                                                        image_dataset["q_test"],
                                                        image_dataset["q_true_dataloader"],
                                                        timesteps,
                                                        config.odeint_batch_size,
                                                        config.ode_solver,
                                                        device)
        
        print("test FID of the particle optimization: {}".format(particle_test_fid))
        print("test FID of the velocity field: {}".format(vf_test_fid))
        print("reconstruction test FID of the particle optimization: {}".format(particle_rec_test_fid))
        print("reconstruction test FID of the velocity field: {}".format(vf_rec_test_fid))
        print("FID of the velocity field vs. particle optimization: {}".format(vf_particle_fid))

        if i == 0:
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "initial_classifier_loss_record" : initial_classifier_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "intermediate_classifier_loss_record" : intermediate_classifier_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record,
                              "vf_test_fid" : vf_test_fid,
                              "particle_test_fid" : particle_test_fid,
                              "vf_particle_fid" : vf_particle_fid}
        else:                
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "intermediate_classifier_loss_record" : intermediate_classifier_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record,
                              "vf_test_fid" : vf_test_fid,
                              "particle_test_fid" : particle_test_fid,
                              "vf_particle_fid" : vf_particle_fid}
            
        save_data(config.saving_dir + "loss_record.pkl", loss_record)
        torch.save(classifier.state_dict(), os.path.join(epoch_saving_dir, 'classifier_e{}.pt'.format(i+1)))
        torch.save(velocity_field.state_dict(), os.path.join(epoch_saving_dir, 'velocity_field_e{}.pt'.format(i+1)))
    
