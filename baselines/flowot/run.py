from baselines.flowot.config import FlowOTImageConfig
from ldm.util import instantiate_from_config
from baselines.flowot.utils import load_data, save_data
from baselines.flowot.utils import DataLoaderIterator, plot_losses, batched_odeint, compute_fid
from baselines.flowot.model.classifier import UNetClassifier
from baselines.flowot.model.velocity_field import ConvVelocityField
from baselines.flowot.model.loss import compute_logit_loss, compute_2wasserstein_loss
from utils import load_image_dataset
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


def run_flowot_image(config: FlowOTImageConfig, dataset_config, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "dataset_config.pkl", dataset_config)
    save_data(config.saving_dir + "flowot_config.pkl", config)

    image_dataset = load_image_dataset(config, dataset_config)

    # load trained VAE
    print("loading pre-trained autoencoder...")
    vae_config = OmegaConf.load(config.vae_config_dir)
    vae_model = instantiate_from_config(vae_config.model)
    vae_model.load_state_dict(torch.load(config.vae_model_dir)["state_dict"], strict=False)

    # no need to load VAE here for training
    timesteps = torch.linspace(0, 1, config.num_timesteps)
    timesteps_reversed = torch.linspace(1, 0, config.num_timesteps)
    input_dim = image_dataset["p_training"][0].shape

    # initialize model
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

    # initialization of the velocity field
    if config.velocity_field_initialization:
        print("Initialize particle trajectories using {}".format(config.velocity_field_initialization))

        vf_init_optim = torch.optim.Adam(velocity_field.parameters(), 
                                        lr=config.velocity_field_learning_rate)
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

    # flow OT refinement
    loss_record = dict()
    initial_pq_classifier_loss_record = []
    initial_qp_classifier_loss_record = []

    kl_loss_pq_record = []
    wasserstein_loss_pq_record = []
    pq_refinement_loss_record = []
    pq_intermediate_classifier_loss_record = []

    kl_loss_qp_record = []
    wasserstein_loss_qp_record = []
    qp_refinement_loss_record = []
    qp_intermediate_classifier_loss_record = []

    for e in range(config.num_iter):

        # training classifier_pq at the first epoch
        if e == 0:
            print("training classifier pq at the first epoch...")
            with torch.no_grad():  # Don't track gradients for ODE solving
                if config.odeint_batch_size:
                    x_p_trajectory = batched_odeint(velocity_field, 
                                                     image_dataset["p_training"], 
                                                     timesteps,
                                                     config.odeint_batch_size,
                                                     ode_solver=config.ode_solver,
                                                     device=device)
                else:
                    x_p_trajectory = odeint(velocity_field, 
                                             image_dataset["p_training"].to(device), 
                                             timesteps.to(device), 
                                             method=config.ode_solver)
                    
            x_p1 = x_p_trajectory[-1,:,:,:,:] # (sample_size, dims)
            classifier_pq_dataloader = DataLoaderIterator(DataLoader(TensorDataset(x_p1, image_dataset["q_training"]), 
                                                            batch_size=config.classifier_training_batch_size, 
                                                            shuffle=True))
            
            classifier_pq_pbar = tqdm(total=config.initial_classifier_training_step,
                                    desc="Initial Classifier PQ Training Steps")

            for x_p1_batch, x_q_batch in classifier_pq_dataloader:
                x_p1_batch = x_p1_batch.to(device)
                x_q_batch = x_q_batch.to(device)
                if classifier_pq_dataloader.step <= config.initial_classifier_training_step:
                    classifier_pq_loss = compute_logit_loss(classifier_pq, x_p1_batch, x_q_batch)
                    classifier_pq_optim.zero_grad()
                    classifier_pq_loss.backward()
                    classifier_pq_optim.step()
                    initial_pq_classifier_loss_record.append(classifier_pq_loss.item())
                    classifier_pq_pbar.update(1)
                else:
                    break

        # training velocity field in forward direction
        print("training velocity field in the forward direction...")
        ckpt_step = 0 # step count for checkpoint
        for i in tqdm(range(config.velocity_field_training_step)):
            ckpt_step += 1

            batch_idx = torch.randperm(image_dataset["p_training"].shape[0])[:config.velocity_field_training_batch_size]
            # need gradient for solving ODE here
            x_p_trajectory_batch = odeint(velocity_field, 
                                          image_dataset["p_training"][batch_idx].to(device), 
                                          timesteps.to(device), 
                                          method=config.ode_solver)
            # permute to (batch_size, len(timesteps), dims, ...)
            x_p_trajectory_batch = x_p_trajectory_batch.permute(1,0,2,3,4).to(device)
            kl_loss_pq = -classifier_pq(x_p_trajectory_batch[:,-1,:,:,:]).mean()
            wasserstein_loss_pq = compute_2wasserstein_loss(x_p_trajectory_batch.flatten(start_dim=2), 
                                                          timesteps.to(device))
            pq_refinement_loss = kl_loss_pq + config.wasserstein_loss_weight * wasserstein_loss_pq
            vf_optim.zero_grad()
            pq_refinement_loss.backward()
            vf_optim.step()

            kl_loss_pq_record.append(kl_loss_pq.item())
            wasserstein_loss_pq_record.append(wasserstein_loss_pq.item())
            pq_refinement_loss_record.append(pq_refinement_loss.item())

            # Update classifier every freq_update epochs
            if (i + 1) % config.classifier_intermediate_training_frequency == 0:
                print("intermediate training of classifier")
                for j in tqdm(range(config.intermediate_classifier_training_step)):
                    
                    p_batch_idx = torch.randperm(image_dataset["p_training"].shape[0])[:config.classifier_training_batch_size]
                    q_batch_idx = torch.randperm(image_dataset["q_training"].shape[0])[:config.classifier_training_batch_size]
  
                    with torch.no_grad():  # don't track gradients for ODE solving
                        x_p_trajectory_batch = odeint(velocity_field, 
                                                        image_dataset["p_training"][p_batch_idx].to(device), 
                                                        timesteps.to(device), 
                                                        method=config.ode_solver)
        
                    x_p1_batch = x_p_trajectory_batch[-1,:,:,:,:].to(device) # (sample_size, dims)
                    x_q_batch = image_dataset["q_training"][q_batch_idx].to(device)
                    classifier_pq_loss = compute_logit_loss(classifier_pq, x_p1_batch, x_q_batch)
                    classifier_pq_optim.zero_grad()
                    classifier_pq_loss.backward()
                    classifier_pq_optim.step()

                    pq_intermediate_classifier_loss_record.append(classifier_pq_loss.item())

            if (ckpt_step+1) % config.checkpoint_step == 0:

                plot_losses(kl_loss_pq_record, 
                            wasserstein_loss_pq_record, 
                            pq_refinement_loss_record,
                            config.saving_dir,
                            "loss_PQ_log_e{}_step{}.pdf".format(e+1, ckpt_step))
                
                # compute FID
                # randomly select the same number of samples in p_test from p_training
                batch_idx = torch.randperm(image_dataset["p_training"].shape[0])[:15000]
                pq_fid = compute_fid(velocity_field, 
                                     vae_model,
                                     image_dataset["q_rescale_factor"],  
                                     image_dataset["p_training"][batch_idx],
                                     image_dataset["q_true_dataloader"],
                                     timesteps,
                                     config.odeint_batch_size,
                                     config.ode_solver,
                                     device)
                
                # randomly select the same number of samples in q_test from q_training
                batch_idx = torch.randperm(image_dataset["q_training"].shape[0])[:15000]
                qp_fid = compute_fid(velocity_field, 
                                     vae_model,
                                     image_dataset["p_rescale_factor"],  
                                     image_dataset["q_training"][batch_idx],
                                     image_dataset["p_true_dataloader"],
                                     timesteps_reversed,
                                     config.odeint_batch_size,
                                     config.ode_solver,
                                     device)

                print("P to Q FID on traslated training samples vs. held-out test samples: {}".format(pq_fid))
                print("Q to P FID on traslated training samples vs. held-out test samples: {}".format(qp_fid))

                loss_record[e] = {"initial_pq_classifier_loss_record" : initial_pq_classifier_loss_record,
                                "initial_qp_classifier_loss_record" : initial_qp_classifier_loss_record,
                                "kl_loss_pq_record" : kl_loss_pq_record,
                                "wasserstein_loss_pq_record" : wasserstein_loss_pq_record,
                                "pq_refinement_loss_record" : pq_refinement_loss_record,
                                "pq_intermediate_classifier_loss_record" : pq_intermediate_classifier_loss_record,
                                "kl_loss_qp_record" : kl_loss_qp_record,
                                "wasserstein_loss_qp_record" : wasserstein_loss_qp_record,
                                "qp_refinement_loss_record" : qp_refinement_loss_record,
                                "qp_intermediate_classifier_loss_record" : qp_intermediate_classifier_loss_record}

                save_data(os.path.join(config.saving_dir, 'loss_record.pkl'), loss_record)
                torch.save(velocity_field.state_dict(), 
                           os.path.join(config.saving_dir, 'velocity_field_pq_e{}_step{}.pt'.format(e+1, ckpt_step)))
                

        if e == 0:

            # training classifier_qp at the first epoch
            print("training classifier qp at the first epoch...")
            with torch.no_grad():  # Don't track gradients for ODE solving
                if config.odeint_batch_size:
                    x_qp_trajectory = batched_odeint(velocity_field, 
                                                     image_dataset["q_training"], 
                                                     timesteps_reversed,
                                                     config.odeint_batch_size,
                                                     ode_solver=config.ode_solver,
                                                     device=device)
                else:
                    x_qp_trajectory = odeint(velocity_field, 
                                             image_dataset["q_training"].to(device), 
                                             timesteps_reversed.to(device), 
                                             method=config.ode_solver) 

            x_q1 = x_qp_trajectory[-1,:,:,:,:] # (sample_size, input_dim)
            classifier_qp_dataloader = DataLoaderIterator(DataLoader(TensorDataset(image_dataset["p_training"], x_q1), 
                                                    batch_size=config.classifier_training_batch_size, 
                                                    shuffle=True))
            
            classifier_qp_pbar = tqdm(total=config.initial_classifier_training_step,
                                        desc="Initial Classifier QP Training Steps")

            for x_p_batch, x_q1_batch in classifier_qp_dataloader:
                x_p_batch = x_p_batch.to(device)
                x_q1_batch = x_q1_batch.to(device)
                if classifier_qp_dataloader.step <= config.initial_classifier_training_step:
                    classifier_qp_loss = compute_logit_loss(classifier_qp, x_q1_batch, x_p_batch)
                    classifier_qp_optim.zero_grad()
                    classifier_qp_loss.backward()
                    classifier_qp_optim.step()
                    initial_qp_classifier_loss_record.append(classifier_qp_loss.item())
                    classifier_qp_pbar.update(1)
                else:
                    break

        # training velocity field in backward direction
        print("training velocity field in the backward direction...")
        ckpt_step = 0
        for i in tqdm(range(config.velocity_field_training_step)):
            ckpt_step += 1

            batch_idx = torch.randperm(image_dataset["q_training"].shape[0])[:config.velocity_field_training_batch_size]
            # need gradient for ODE solving here
            x_q_trajectory_batch = odeint(velocity_field, 
                                          image_dataset["q_training"][batch_idx].to(device), 
                                          timesteps_reversed.to(device), 
                                          method=config.ode_solver)
            # (batch_size, len(timesteps), dims, ...)
            x_q_trajectory_batch = x_q_trajectory_batch.permute(1,0,2,3,4).to(device)
            kl_loss_qp = -classifier_qp(x_q_trajectory_batch[:,-1,:,:,:]).mean()
            wasserstein_loss_qp = compute_2wasserstein_loss(x_q_trajectory_batch.flatten(start_dim=2), 
                                                          timesteps_reversed.to(device))
            qp_refinement_loss = kl_loss_qp + config.wasserstein_loss_weight * wasserstein_loss_qp
            vf_optim.zero_grad()
            qp_refinement_loss.backward()
            vf_optim.step()

            kl_loss_qp_record.append(kl_loss_qp.item())
            wasserstein_loss_qp_record.append(wasserstein_loss_qp.item())
            qp_refinement_loss_record.append(qp_refinement_loss.item())

            # Update classifier every freq_update epochs
            if (i + 1) % config.classifier_intermediate_training_frequency == 0:
                print("intermediate training of classifier")
                for j in tqdm(range(config.intermediate_classifier_training_step)):
                    
                    batch_idx = torch.randperm(image_dataset["q_training"].shape[0])[:config.classifier_training_batch_size]
                    
                    with torch.no_grad():  # don't track gradients for ODE solving
                        x_q_trajectory_batch = odeint(velocity_field, 
                                                      image_dataset["q_training"][batch_idx].to(device), 
                                                      timesteps_reversed.to(device), 
                                                      method=config.ode_solver)
        
                    x_q1_batch = x_q_trajectory_batch[-1,:,:,:,:].to(device) # (sample_size, input_dim)
                    x_p_batch = image_dataset["p_training"][batch_idx].to(device)
                    classifier_qp_loss = compute_logit_loss(classifier_qp, x_q1_batch, x_p_batch)
                    classifier_qp_optim.zero_grad()
                    classifier_qp_loss.backward()
                    classifier_qp_optim.step()
                    qp_intermediate_classifier_loss_record.append(classifier_qp_loss.item())

            if (ckpt_step+1) % config.checkpoint_step == 0:

                plot_losses(kl_loss_qp_record, 
                            wasserstein_loss_qp_record, 
                            qp_refinement_loss_record,
                            config.saving_dir,
                            "loss_QP_log_e{}_step{}.pdf".format(e+1, ckpt_step))
                
                # compute FID

                batch_idx = torch.randperm(image_dataset["p_training"].shape[0])[:15000]
                pq_fid = compute_fid(velocity_field, 
                                     vae_model,
                                     image_dataset["q_rescale_factor"],  
                                     image_dataset["p_training"][batch_idx],
                                     image_dataset["q_true_dataloader"],
                                     timesteps,
                                     config.odeint_batch_size,
                                     config.ode_solver,
                                     device)
                
                # randomly select the same number of samples in q_test from q_training
                batch_idx = torch.randperm(image_dataset["q_training"].shape[0])[:15000]
                qp_fid = compute_fid(velocity_field, 
                                     vae_model,
                                     image_dataset["p_rescale_factor"],  
                                     image_dataset["q_training"][batch_idx],
                                     image_dataset["p_true_dataloader"],
                                     timesteps_reversed,
                                     config.odeint_batch_size,
                                     config.ode_solver,
                                     device)

                print("P to Q FID on traslated training samples vs. held-out test samples: {}".format(pq_fid))
                print("Q to P FID on traslated training samples vs. held-out test samples: {}".format(qp_fid))

                loss_record[e] = {"initial_pq_classifier_loss_record" : initial_pq_classifier_loss_record,
                                "initial_qp_classifier_loss_record" : initial_qp_classifier_loss_record,
                                "kl_loss_pq_record" : kl_loss_pq_record,
                                "wasserstein_loss_pq_record" : wasserstein_loss_pq_record,
                                "pq_refinement_loss_record" : pq_refinement_loss_record,
                                "pq_intermediate_classifier_loss_record" : pq_intermediate_classifier_loss_record,
                                "kl_loss_qp_record" : kl_loss_qp_record,
                                "wasserstein_loss_qp_record" : wasserstein_loss_qp_record,
                                "qp_refinement_loss_record" : qp_refinement_loss_record,
                                "qp_intermediate_classifier_loss_record" : qp_intermediate_classifier_loss_record}

                save_data(os.path.join(config.saving_dir, 'loss_record.pkl'), loss_record)
                torch.save(velocity_field.state_dict(), 
                           os.path.join(config.saving_dir, 'velocity_field_pq_e{}_step{}.pt'.format(e+1, ckpt_step)))
                
        loss_record[e] = {"initial_pq_classifier_loss_record" : initial_pq_classifier_loss_record,
                          "initial_qp_classifier_loss_record" : initial_qp_classifier_loss_record,
                          "kl_loss_pq_record" : kl_loss_pq_record,
                          "wasserstein_loss_pq_record" : wasserstein_loss_pq_record,
                          "pq_refinement_loss_record" : pq_refinement_loss_record,
                          "pq_intermediate_classifier_loss_record" : pq_intermediate_classifier_loss_record,
                          "kl_loss_qp_record" : kl_loss_qp_record,
                          "wasserstein_loss_qp_record" : wasserstein_loss_qp_record,
                          "qp_refinement_loss_record" : qp_refinement_loss_record,
                          "qp_intermediate_classifier_loss_record" : qp_intermediate_classifier_loss_record}

        save_data(os.path.join(config.saving_dir, 'loss_record.pkl'), loss_record)
        torch.save(velocity_field.state_dict(), 
                   os.path.join(config.saving_dir, 'velocity_field_e{}.pt'.format(e+1)))

