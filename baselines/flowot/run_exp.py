from baselines.flowot.config import FlowOTConfig
from baselines.flowot.config import ShoebagsDatasetConfig, CelebADatasetConfig
from ldm.models.utils import autoencoder_decode, convert_logvar_to_std
from ldm.data.shoebags import Shoes, Bags
from ldm.data.base import ImageDefaultDataset
from ldm.util import instantiate_from_config
from baselines.flowot.utils import load_data, save_data, plot_losses
from baselines.flowot.utils import compute_rescale_factor, generate_latent_image_data, load_shoebags_latent_params
from baselines.flowot.model.classifier import UNetClassifier
from baselines.flowot.model.velocity_field import ConvVelocityField
from baselines.flowot.model.loss import compute_logit_loss, compute_2wassertain_loss
from baselines.torchcfm.conditional_flow_matching import VariancePreservingConditionalFlowMatcher
from utils import DataLoaderIterator
from utils import batched_odeint
from utils import visualize_decoded_samples_trajectories, compute_fid
from omegaconf import OmegaConf
from torchdiffeq import odeint
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os


def run_flowot_image(config: FlowOTConfig, dataset_config, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "dataset_config.pkl", dataset_config)
    save_data(config.saving_dir + "flowot_config.pkl", config)

    if isinstance(dataset_config, ShoebagsDatasetConfig):

        image_dim = (64, 64, 3)
        shoes_mu, shoes_logvar, bags_mu, bags_logvar = load_shoebags_latent_params(dataset_config)
        shoes_std = convert_logvar_to_std(shoes_logvar)
        bags_std = convert_logvar_to_std(bags_logvar)
        q_rescale_factor = compute_rescale_factor(shoes_mu, shoes_std)
        p_rescale_factor = compute_rescale_factor(bags_mu, bags_std)
        p_training, p_test, q_training, q_test = generate_latent_image_data(bags_mu, bags_std, # P
                                                                            shoes_mu, shoes_std, # Q
                                                                            dataset_config.train_ratio,
                                                                            dataset_config.seed,
                                                                            p_rescale_factor, 
                                                                            q_rescale_factor)
        
        # load shoebags image dataset
        p_test_dataset = Bags(dataset_config.bags_dataset_dir, 
                              dataset_config.train_ratio, 
                              dataset_config.seed, 
                              complement=True)
        q_test_dataset = Shoes(dataset_config.shoes_dataset_dir, 
                               dataset_config.train_ratio, 
                               dataset_config.seed, 
                               complement=True)
        p_test_dataloader = DataLoader(ImageDefaultDataset(p_test_dataset), 
                                        batch_size=config.odeint_batch_size, 
                                        shuffle=False)
        q_test_dataloader = DataLoader(ImageDefaultDataset(q_test_dataset), 
                                        batch_size=config.odeint_batch_size, 
                                        shuffle=False)
        
    elif isinstance(dataset_config, CelebADatasetConfig):
        # TODO
        pass

    # load trained VAE
    print("loading pre-trained autoencoder...")
    vae_config = OmegaConf.load(config.vae_config_dir)
    vae_model = instantiate_from_config(vae_config.model)
    vae_model.load_state_dict(torch.load(config.vae_model_dir)["state_dict"], strict=False)

    timesteps = torch.linspace(0, 1, config.num_timesteps).to(device)
    timesteps_reversed = torch.linspace(1, 0, config.num_timesteps).to(device)
    input_dim = p_training[0].shape

    # initialize model
    print("data dimension: {}".format(input_dim))
    print("training data size of P : {}".format(p_training.shape))
    print("training data size of Q : {}".format(q_training.shape))

    classifier_pq = UNetClassifier(input_dim[0], 
                                config.classifier_channels,
                                config.classifier_use_bias).to(device)
    classifier_qp = UNetClassifier(input_dim[0], 
                                config.classifier_channels,
                                config.classifier_use_bias).to(device)
    velocity_field = ConvVelocityField(input_dim[0], 
                                      config.velocity_field_encoding_dims,
                                      config.velocity_field_decoding_dims,
                                      config.velocity_field_kernel_sizes,
                                      config.velocity_field_strides).to(device)
    
    # optimizer
    classifier_pq_optim = torch.optim.Adam(classifier_pq.parameters(), lr=config.classifier_learning_rate)
    classifier_qp_optim = torch.optim.Adam(classifier_qp.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.velocity_field_learning_rate)

    if config.initial_velocity_field_dir:
        # TODO
        pass

    else:
        print("Initialize with flow matching on stochastic interpolant")
        vf_init_optim = torch.optim.Adam(velocity_field.parameters(), 
                                         lr=config.velocity_field_learning_rate)
        # dataloader should use entire data, not a part of the data
        vf_init_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_training, q_training), 
                                                            batch_size=config.velocity_field_training_batch_size, 
                                                            shuffle=True))
        vf_init_pbar = tqdm(total=config.velocity_field_initial_training_step, 
                            desc="Velocity Fields Initialization")
        
        flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
    
        for p_batch, q_batch in vf_init_dataloader:
            if vf_init_dataloader.step <= config.velocity_field_initial_training_step:
                t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, q_batch)

                t_batch = t_batch.to(device)
                xt_batch = xt_batch.to(device)
                ut_batch = ut_batch.to(device)
                vt_batch = velocity_field(t_batch, xt_batch)
                loss = torch.mean((vt_batch - ut_batch) ** 2)
                vf_init_optim.zero_grad()
                loss.backward()
                vf_init_optim.step()
                vf_init_pbar.update(1)
            else:
                break

    # flow OT refinement
    loss_record = dict()
    initial_pq_classifier_loss_record = []
    initial_qp_classifier_loss_record = []

    kl_loss_pq_record = []
    wassertain_loss_pq_record = []
    pq_refinement_loss_record = []
    pq_intermediate_classifier_loss_record = []

    kl_loss_qp_record = []
    wassertain_loss_qp_record = []
    qp_refinement_loss_record = []
    qp_intermediate_classifier_loss_record = []

    # training classifier_pq at the first epoch
    print("training classifier pq at the first epoch...")
    with torch.no_grad():  # Don't track gradients for ODE solving
        if config.odeint_batch_size:
            x_pq_trajectory = batched_odeint(velocity_field, 
                                             p_training, 
                                             timesteps,
                                             config.odeint_batch_size,
                                             ode_solver=config.ode_solver,
                                             device=device)
        else:
            x_pq_trajectory = odeint(velocity_field, 
                                     p_training, 
                                     timesteps, 
                                     method=config.ode_solver)
    x_pq = x_pq_trajectory[-1,:,:,:,:] # (sample_size, dims)
    classifier_pq_dataloader = DataLoaderIterator(DataLoader(TensorDataset(x_pq, q_training), 
                                                    batch_size=config.classifier_training_batch_size, 
                                                    shuffle=True))
    
    classifier_pq_pbar = tqdm(total=config.initial_classifier_training_step,
                                desc="Initial Classifier PQ Training Steps")

    for x_pq_batch, x_q_batch in classifier_pq_dataloader:
        x_pq_batch = x_pq_batch.to(device)
        x_q_batch = x_q_batch.to(device)
        if classifier_pq_dataloader.step <= config.initial_classifier_training_step:
            classifier_pq_loss = compute_logit_loss(classifier_pq, x_pq_batch, x_q_batch)
            classifier_pq_optim.zero_grad()
            classifier_pq_loss.backward()
            classifier_pq_optim.step()
            initial_pq_classifier_loss_record.append(classifier_pq_loss.item())
            classifier_pq_pbar.update(1)
        else:
            break

    # training velocity field in forward direction
    print("training velocity field in the forward direction...")
    for i in tqdm(range(config.velocity_field_training_step)):

        batch_idx = torch.randperm(p_training.shape[0])[:config.velocity_field_training_batch_size]
        # need gradient for ODE solving here
        x_p_trajectory_batch = odeint(velocity_field, 
                                        p_training[batch_idx].to(device), 
                                        timesteps, 
                                        method=config.ode_solver)
        # (batch_size, len(timesteps), dims, ...)
        x_p_trajectory_batch = x_p_trajectory_batch.permute(1,0,2,3,4).to(device)
        kl_loss_pq = -classifier_pq(x_p_trajectory_batch[:,-1,:,:,:]).mean()
        wassertain_loss_pq = compute_2wassertain_loss(x_p_trajectory_batch.flatten(start_dim=2), 
                                                        timesteps)
        pq_refinement_loss = kl_loss_pq + config.wassertain_loss_weight * wassertain_loss_pq
        vf_optim.zero_grad()
        pq_refinement_loss.backward()
        vf_optim.step()

        kl_loss_pq_record.append(kl_loss_pq.item())
        wassertain_loss_pq_record.append(wassertain_loss_pq.item())
        pq_refinement_loss_record.append(pq_refinement_loss.item())

        # Update classifier every freq_update epochs
        if (i + 1) % config.classifier_intermediate_training_frequency == 0:
            print("intermediate training of classifier")
            for j in tqdm(range(config.intermediate_classifier_training_step)):
                
                batch_idx = torch.randperm(p_training.shape[0])[:config.classifier_training_batch_size]
                
                with torch.no_grad():  # don't track gradients for ODE solving
                    x_p_trajectory_batch = odeint(velocity_field, 
                                                    p_training[batch_idx].to(device), 
                                                    timesteps, 
                                                    method=config.ode_solver)
    
                x_pq_batch = x_p_trajectory_batch[-1,:,:,:,:].to(device) # (sample_size, input_dim)
                x_q_batch = q_training[batch_idx].to(device)
                classifier_pq_loss = compute_logit_loss(classifier_pq, x_pq_batch, x_q_batch)
                classifier_pq_optim.zero_grad()
                classifier_pq_loss.backward()
                classifier_pq_optim.step()

                pq_intermediate_classifier_loss_record.append(classifier_pq_loss.item())

        if (i+1) % config.checkpoint_step == 0:
            # checkpoint: plot loss, compute FID, save models

            plot_losses(kl_loss_pq_record, 
                        wassertain_loss_pq_record, 
                        pq_refinement_loss_record,
                        config.saving_dir,
                        "loss_PQ_log.pdf")
            
            # compute FID between generated from training data vs. test held-out set
            p_test_size = len(p_test_dataset)
            q_test_size = len(q_test_dataset)
            p_batch_idx = torch.randperm(p_training.shape[0])[:p_test_size]
            q_batch_idx = torch.randperm(q_training.shape[0])[:q_test_size]
            p_selected_batch = p_training[p_batch_idx]
            q_selected_batch = q_training[q_batch_idx]

            pq_fid = compute_fid(velocity_field, 
                                 vae_model, 
                                 q_rescale_factor, 
                                 p_selected_batch, 
                                 q_test_dataloader,
                                 timesteps, 
                                 config.odeint_batch_size, 
                                 config.ode_solver, 
                                 device=device)
            
            qp_fid = compute_fid(velocity_field, 
                                 vae_model, 
                                 p_rescale_factor, 
                                 q_selected_batch, 
                                 p_test_dataloader,
                                 timesteps_reversed, 
                                 config.odeint_batch_size, 
                                 config.ode_solver, 
                                 device=device)

            print("P to Q FID on traslated samples vs. held-out test samples: {}".format(pq_fid))
            print("Q to P FID on traslated samples vs. held-out test samples: {}".format(qp_fid))

            torch.save(velocity_field.state_dict(), 
                       os.path.join(config.saving_dir, 'velocity_field_pq_step{}.pt'.format(i+1)))


    # training classifier_qp at the first epoch
    print("training classifier qp at the first epoch...")
    with torch.no_grad():  # Don't track gradients for ODE solving
        if config.odeint_batch_size:
            x_qp_trajectory = batched_odeint(velocity_field, 
                                                q_training, 
                                                timesteps_reversed,
                                                config.odeint_batch_size,
                                                ode_solver=config.ode_solver,
                                                device=device)
        else:
            x_qp_trajectory = odeint(velocity_field, 
                                        q_training.to(device), 
                                        timesteps_reversed, 
                                        method=config.ode_solver) 

    x_qp = x_qp_trajectory[-1,:,:,:,:] # (sample_size, input_dim)
    classifier_qp_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_training, x_qp), 
                                            batch_size=config.classifier_training_batch_size, 
                                            shuffle=True))
    
    classifier_qp_pbar = tqdm(total=config.initial_classifier_training_step,
                                desc="Initial Classifier QP Training Steps")

    for x_p_batch, x_qp_batch in classifier_qp_dataloader:
        x_p_batch = x_p_batch.to(device)
        x_qp_batch = x_qp_batch.to(device)
        if classifier_qp_dataloader.step <= config.initial_classifier_training_step:
            classifier_qp_loss = compute_logit_loss(classifier_qp, x_qp_batch, x_p_batch)
            classifier_qp_optim.zero_grad()
            classifier_qp_loss.backward()
            classifier_qp_optim.step()
            initial_qp_classifier_loss_record.append(classifier_qp_loss.item())
            classifier_qp_pbar.update(1)
        else:
            break

    # training velocity field in backward direction
    print("training velocity field in the backward direction...")
    for i in tqdm(range(config.velocity_field_training_step)):

        batch_idx = torch.randperm(q_training.shape[0])[:config.velocity_field_training_batch_size]
        # need gradient for ODE solving here
        x_q_trajectory_batch = odeint(velocity_field, 
                                      q_training[batch_idx].to(device), 
                                      timesteps_reversed, 
                                      method=config.ode_solver)
        # (batch_size, len(timesteps), dims, ...)
        x_q_trajectory_batch = x_q_trajectory_batch.permute(1,0,2,3,4).to(device)
        kl_loss_qp = -classifier_qp(x_q_trajectory_batch[:,-1,:,:,:]).mean()
        wassertain_loss_qp = compute_2wassertain_loss(x_q_trajectory_batch.flatten(start_dim=2), 
                                                        timesteps_reversed)
        qp_refinement_loss = kl_loss_qp + config.wassertain_loss_weight * wassertain_loss_qp
        vf_optim.zero_grad()
        qp_refinement_loss.backward()
        vf_optim.step()

        kl_loss_qp_record.append(kl_loss_qp.item())
        wassertain_loss_qp_record.append(wassertain_loss_qp.item())
        qp_refinement_loss_record.append(qp_refinement_loss.item())

        # Update classifier every freq_update epochs
        if (i + 1) % config.classifier_intermediate_training_frequency == 0:
            print("intermediate training of classifier")
            for j in tqdm(range(config.intermediate_classifier_training_step)):
                
                batch_idx = torch.randperm(q_training.shape[0])[:config.classifier_training_batch_size]
                
                with torch.no_grad():  # don't track gradients for ODE solving
                    x_q_trajectory_batch = odeint(velocity_field, 
                                                    q_training[batch_idx].to(device), 
                                                    timesteps_reversed, 
                                                    method=config.ode_solver)
    
                x_qp_batch = x_q_trajectory_batch[-1,:,:,:,:].to(device) # (sample_size, input_dim)
                x_p_batch = p_training[batch_idx].to(device)
                classifier_qp_loss = compute_logit_loss(classifier_qp, x_qp_batch, x_p_batch)
                classifier_qp_optim.zero_grad()
                classifier_qp_loss.backward()
                classifier_qp_optim.step()
                qp_intermediate_classifier_loss_record.append(classifier_qp_loss.item())


        if (i+1) % config.checkpoint_step == 0:
            # checkpoint: plot loss, compute FID, save models

            plot_losses(kl_loss_qp_record, 
                        wassertain_loss_qp_record, 
                        qp_refinement_loss_record,
                        config.saving_dir,
                        "loss_QP_log.pdf")
            
            # compute FID between generated from training data vs. test held-out set
            p_test_size = len(p_test_dataset)
            q_test_size = len(q_test_dataset)
            p_batch_idx = torch.randperm(p_training.shape[0])[:p_test_size]
            q_batch_idx = torch.randperm(q_training.shape[0])[:q_test_size]
            p_selected_batch = p_training[p_batch_idx]
            q_selected_batch = q_training[q_batch_idx]

            pq_fid = compute_fid(velocity_field, 
                                 vae_model, 
                                 q_rescale_factor, 
                                 p_selected_batch, 
                                 q_test_dataloader,
                                 timesteps, 
                                 config.odeint_batch_size, 
                                 config.ode_solver, 
                                 device=device)
            
            qp_fid = compute_fid(velocity_field, 
                                 vae_model, 
                                 p_rescale_factor, 
                                 q_selected_batch, 
                                 p_test_dataloader,
                                 timesteps_reversed, 
                                 config.odeint_batch_size, 
                                 config.ode_solver, 
                                 device=device)

            print("P to Q FID on traslated samples vs. held-out test samples: {}".format(pq_fid))
            print("Q to P FID on traslated samples vs. held-out test samples: {}".format(qp_fid))

            torch.save(velocity_field.state_dict(), 
                       os.path.join(config.saving_dir, 'velocity_field_qp_step{}.pt'.format(i+1)))

    # select 5 images for sanity check
    pq_rand_idx_visualization = torch.randperm(x_pq_trajectory.shape[1])[:5]
    qp_rand_idx_visualization = torch.randperm(x_qp_trajectory.shape[1])[:5]

    # (num_timesteps, len(rand_idx_visualization), channel, w, h)
    x_pq_visualization_samples = x_pq_trajectory[:,pq_rand_idx_visualization,:,:,:].flatten(0,1)
    vae_model.eval()
    pq_decoded_samples = autoencoder_decode(vae_model,
                                            x_pq_visualization_samples, 
                                            rescale_factor=q_rescale_factor,
                                            batch_size=None,
                                            device=device)
    pq_decoded_samples = pq_decoded_samples.view((len(timesteps), 
                                            len(pq_rand_idx_visualization),
                                            image_dim[2],
                                            image_dim[0],
                                            image_dim[1])).cpu().detach()
        
    x_qp_visualization_samples = x_qp_trajectory[:,qp_rand_idx_visualization,:,:,:].flatten(0,1)
    vae_model.eval()
    qp_decoded_samples = autoencoder_decode(vae_model,
                                            x_qp_visualization_samples, 
                                            rescale_factor=q_rescale_factor,
                                            batch_size=None,
                                            device=device)
    qp_decoded_samples = qp_decoded_samples.view((len(timesteps), 
                                            len(pq_rand_idx_visualization),
                                            image_dim[2],
                                            image_dim[0],
                                            image_dim[1])).cpu().detach()
        
    pq_saving_file_name = os.path.join(config.saving_dir, "5decoded_pq_samples_trajectories.pdf")
    qp_saving_file_name = os.path.join(config.saving_dir, "5decoded_qp_samples_trajectories.pdf")
    visualize_decoded_samples_trajectories(pq_decoded_samples, 8, pq_saving_file_name)
    visualize_decoded_samples_trajectories(qp_decoded_samples, 8, qp_saving_file_name)

    loss_record = {"kl_loss_pq_record" : kl_loss_pq_record,
                        "wassertain_loss_pq_record" : wassertain_loss_pq_record,
                        "pq_refinement_loss_record" : pq_refinement_loss_record,
                        "kl_loss_qp_record" : kl_loss_qp_record,
                        "pq_intermediate_classifier_loss_record" : pq_intermediate_classifier_loss_record,
                        "wassertain_loss_qp_record" : wassertain_loss_qp_record,
                        "qp_refinement_loss_record" : qp_refinement_loss_record,
                        "qp_intermediate_classifier_loss_record" : qp_intermediate_classifier_loss_record}
    
    # save loss record every epoch
    save_data(config.saving_dir + "loss_record.pkl", loss_record)
    torch.save(classifier_pq.state_dict(), os.path.join(config.saving_dir, 'classifier_pq.pt'))
    torch.save(classifier_qp.state_dict(), os.path.join(config.saving_dir, 'classifier_qp.pt'))
    torch.save(velocity_field.state_dict(), os.path.join(config.saving_dir, 'velocity_field.pt'))

