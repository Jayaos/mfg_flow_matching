from config import MFGFlowImageConfig
from model.classifier import UNetClassifier
from model.velocity_field import ConvVelocityField
from model import classifier_loss_fn, particle_optimization_loss_fn, image_vf_loss_fn
from utils import load_image_dataset, DataLoaderIterator, TrajectoryDataLoader
from utils import initialize_X_bar_image, batched_odeint
from utils import visualize_decoded_samples_trajectories
from utils import load_data, save_data, compute_fid, compute_test_fid
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
    timesteps = torch.linspace(0, 1, config.ode_timesteps)
    timestep_size = 1/(config.ode_timesteps-1)
    input_dim = image_dataset["p_training"][0].shape

    # initialize model
    print("data dimension: {}".format(input_dim))
    print("training data size of P : {}".format(image_dataset["p_training"].shape))
    print("training data size of Q : {}".format(image_dataset["q_training"].shape))

    outer_loop_dataloader = DataLoaderIterator(DataLoader(TensorDataset(image_dataset["p_training"], 
                                                                        image_dataset["q_training"]), 
                                                          batch_size=config.outer_batch, 
                                                          shuffle=True, 
                                                          drop_last=True))

    classifier = UNetClassifier(input_dim[0], 
                                config.classifier_channels,
                                config.classifier_use_bias).to(device)
    velocity_field = ConvVelocityField(input_dim[0], 
                                       config.vf_encoding_dims,
                                       config.vf_decoding_dims,
                                       config.vf_kernel_sizes,
                                       config.vf_strides).to(device)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.vf_learning_rate)

    initial_classifier_loss_record = []
    loss_record = dict()

    mfgflow_training_pbar = tqdm(total=config.outer_loop, 
                                 desc="MFG-Flow Training Epochs")

    for i in range(config.outer_loop):

        epoch_saving_dir = os.path.join(config.saving_dir, "epoch_{}".format(i+1))
        os.makedirs(epoch_saving_dir, exist_ok=True)

        p_training_loop, q_training_loop = next(outer_loop_dataloader)

        if i == 0:

            # on the first loop,
            # initialize velocity field, solve ODE to obtain trajectories, and train classifier

            vf_init_optim = torch.optim.Adam(velocity_field.parameters(), 
                                             lr=config.vf_learning_rate)
            vf_init_dataloader = DataLoaderIterator(DataLoader(TensorDataset(image_dataset["p_training"], 
                                                                             image_dataset["q_training"]), 
                                                                batch_size=config.vf_minibatch, 
                                                                shuffle=True))
            vf_init_pbar = tqdm(total=config.vf_initial_steps, 
                                desc="Velocity Field Initialization")
            
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
                                           p_training_loop, 
                                           timesteps,
                                           config.odeint_minibatch,
                                           ode_solver=config.ode_solver,
                                           device=device)
                else:
                    X_bar = odeint(velocity_field, p_training_loop, timesteps, method=config.ode_solver) 
                # (len(timesteps), training_size, dims, ...)

            # select 5 images for sanity check
            rand_idx_visualization = torch.randperm(len(p_training_loop))[:5]
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
                
            # test FID at initialization
            init_vf_test_fid = compute_test_fid(velocity_field, 
                                                vae_model, 
                                                image_dataset["q_rescale_factor"],
                                                image_dataset["p_test"],
                                                image_dataset["q_test_dataloader"],
                                                timesteps,
                                                config.odeint_minibatch,
                                                config.ode_solver,
                                                device=device)
            print("test FID of the initialized velocity field: {}".format(init_vf_test_fid))

            # initial classifier training
            classifier_optimization_pbar = tqdm(total=config.classifier_initial_steps, 
                                                desc="Initial Classifier Training Steps")
            p_1_training = X_bar[-1,:,:,:,:].detach() # (data_size, dim), endpoint of linear interpolant
            classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training_loop), 
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
                    # config.initial_classifier_training_step reached break the for loop
                    break

        else:
            # if not the first loop, start by solving ODE to obtain trajectories
            with torch.no_grad():  # Don't track gradients for ODE solving
                if config.odeint_minibatch:
                    X_bar = batched_odeint(velocity_field, 
                                           p_training_loop, 
                                           timesteps,
                                           config.odeint_minibatch,
                                           ode_solver=config.ode_solver,
                                           device=device)
                else:
                    X_bar = odeint(velocity_field, p_training_loop, timesteps, method=config.ode_solver) 
                # (len(timesteps), training_size, dims, ...)
                         
            # select 5 images for sanity check
            rand_idx_visualization = torch.randperm(len(p_training_loop))[:5]
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
        particle_optim = torch.optim.Adam([particle_trajectory], lr=config.particle_learning_rate)


        kinetic_loss_record = []
        classifier_loss_record = []
        particle_optimization_loss_record = []
        classifier_retrain_loss_record = []

        particle_dataloader = TrajectoryDataLoader(particle_0, 
                                                   particle_trajectory,
                                                   batch_size=config.particle_minibatch, 
                                                   shuffle=True)
        particle_dataloader_size = len(particle_dataloader)
                
        for e in tqdm(range(config.particle_loop)):   
                             
            kinetic_loss_sum = 0.
            classifier_loss_sum = 0.
            particle_optimization_loss_sum = 0.

            for particle_0_batch, particle_trajectory_batch in particle_dataloader:

                particle_optimization_loss = particle_optimization_loss_fn(particle_0_batch, 
                                                                           particle_trajectory_batch,
                                                                           classifier,
                                                                           kinetic_loss_weight=config.kinetic_loss_weight)
                particle_optim.zero_grad()
                particle_optimization_loss["loss"].backward()
                particle_optim.step()

                kinetic_loss_sum += particle_optimization_loss["kinetic_loss"].item()
                classifier_loss_sum += particle_optimization_loss["classifier_loss"].item()
                particle_optimization_loss_sum += particle_optimization_loss["loss"].item()

            kinetic_loss_record.append(kinetic_loss_sum / particle_dataloader_size)
            classifier_loss_record.append(classifier_loss_sum / particle_dataloader_size)
            particle_optimization_loss_record.append(particle_optimization_loss_sum / particle_dataloader_size)

            # Update classifier every freq_update epochs
            if (e + 1) % config.cost_update_frequency == 0:
                    
                p_1_training = particle_trajectory[-1, :, :].clone().detach() # (data_size, dim)
                classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training_loop), 
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
                        classifier_retrain_loss_record.append(classifier_loss["loss"].item())
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
        particle_train_fid, vf_train_fid, particle_rec_train_fid, \
        vf_rec_train_fid, vf_particle_train_fid = compute_fid(velocity_field, 
                                                        vae_model,
                                                        image_dataset["q_rescale_factor"],  
                                                        X_bar,
                                                        image_dataset["q_test"],
                                                        image_dataset["q_test_dataloader"],
                                                        timesteps,
                                                        config.odeint_minibatch,
                                                        config.ode_solver,
                                                        device)
        
        test_fid = compute_test_fid(velocity_field, 
                                    vae_model, 
                                    image_dataset["q_rescale_factor"],
                                    image_dataset["p_test"],
                                    image_dataset["q_test_dataloader"],
                                    timesteps,
                                    config.odeint_minibatch,
                                    config.ode_solver,
                                    device=device)
        
        print("train FID of the particle optimization: {}".format(particle_train_fid))
        print("train FID of the velocity field: {}".format(vf_train_fid))
        print("reconstruction train FID of the particle optimization: {}".format(particle_rec_train_fid))
        print("reconstruction train FID of the velocity field: {}".format(vf_rec_train_fid))
        print("FID of the velocity field vs. particle optimization: {}".format(vf_particle_train_fid))
        print("TEST FID: {}".format(test_fid))

        if i == 0:
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "initial_classifier_loss_record" : initial_classifier_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "classifier_retrain_loss_record" : classifier_retrain_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record,
                              "vf_train_fid" : vf_train_fid,
                              "particle_train_fid" : particle_train_fid,
                              "vf_particle_train_fid" : vf_particle_train_fid,
                              "particle_rec_train_fid" : particle_rec_train_fid,
                              "vf_rec_train_fid" : vf_rec_train_fid,
                              "test_fid" : test_fid}
        else:                
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "classifier_retrain_loss_record" : classifier_retrain_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record,
                              "vf_train_fid" : vf_train_fid,
                              "particle_train_fid" : particle_train_fid,
                              "vf_particle_train_fid" : vf_particle_train_fid,
                              "particle_rec_train_fid" : particle_rec_train_fid,
                              "vf_rec_train_fid" : vf_rec_train_fid,
                              "test_fid" : test_fid}
            
        save_data(config.saving_dir + "loss_record.pkl", loss_record)
        torch.save(classifier.state_dict(), os.path.join(epoch_saving_dir, 'classifier_l{}.pt'.format(i+1)))
        torch.save(velocity_field.state_dict(), os.path.join(epoch_saving_dir, 'velocity_field_l{}.pt'.format(i+1)))
    
