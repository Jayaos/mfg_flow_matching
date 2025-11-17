from config import BaselineImageConfig
from model.velocity_field import ConvVelocityField
from baselines.torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import VariancePreservingConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from ldm.models.utils import autoencoder_decode
from ldm.util import instantiate_from_config
from utils import DataLoaderIterator
from utils import load_data, load_image_dataset, save_data, compute_fid_endpoint
from utils import visualize_decoded_samples_trajectories, batched_odeint
from omegaconf import OmegaConf
from torchdiffeq import odeint
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os


def run_baselines_image(config: BaselineImageConfig, dataset_config, device):
    """
    run baselines on image tasks
    baselines: OT-CFM, SB-CFM, Stochastic Interpolants, Rectified Flow
    """
    
    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "dataset_config.pkl", dataset_config)
    save_data(config.saving_dir + "baseline_config.pkl", config)

    image_dataset = load_image_dataset(config, dataset_config)

    # load trained VAE
    print("loading pre-trained autoencoder...")
    vae_config = OmegaConf.load(config.vae_config_dir)
    vae_model = instantiate_from_config(vae_config.model)
    vae_model.load_state_dict(torch.load(config.vae_model_dir)["state_dict"], strict=False)

    # no need to load VAE here for training
    timesteps = torch.linspace(0, 1, config.num_timesteps)
    input_dim = image_dataset["p_training"][0].shape

    # initialize model
    print("data dimension: {}".format(input_dim))
    print("data size of P : {}".format(image_dataset["p_training"].shape))
    print("data size of Q : {}".format(image_dataset["q_training"].shape))
    
    velocity_field = ConvVelocityField(input_dim[0], 
                                      config.velocity_field_encoding_dims,
                                      config.velocity_field_decoding_dims,
                                      config.velocity_field_kernel_sizes,
                                      config.velocity_field_strides).to(device)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.learning_rate)

    print("Baseline : {}".format(config.baseline))

    if config.baseline in ["interflow", "stochastic-interpolant"]:
        flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
    elif config.baseline in ["ot-cfm", "otcfm"]:
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
    elif config.baseline in ["sbcfm", "sb-cfm"]:
        flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=0.1, ot_method="exact")
    elif config.baseline in ["reflow", "rectified-flow"]:
        flow_matcher = ConditionalFlowMatcher(sigma=0.)

    loss_record = []
    training_dataloader = DataLoaderIterator(DataLoader(TensorDataset(image_dataset["p_training"], 
                                                                        image_dataset["q_training"]), 
                                                        batch_size=config.training_batch_size, 
                                                        shuffle=True))
    training_pbar = tqdm(total=config.max_training_step, desc="Training Steps")

    for p_batch, q_batch in training_dataloader:

        if training_dataloader.step <= config.max_training_step:
            t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, q_batch)
            t_batch = t_batch.to(device)
            xt_batch = xt_batch.to(device)
            ut_batch = ut_batch.to(device)
            vt_batch = velocity_field(t_batch, xt_batch)
            loss = torch.mean((vt_batch - ut_batch) ** 2)
            vf_optim.zero_grad()
            loss.backward()
            vf_optim.step()
            loss_record.append(loss.item())
            training_pbar.update(1)
        else:
            # config.max_training_step reached break the for loop
            break

        if ((training_dataloader.step+1) % config.checkpoint) == 0:

            ckpt_saving_dir = os.path.join(config.saving_dir, "ckpt_{}".format(training_dataloader.step+1))
            os.makedirs(ckpt_saving_dir, exist_ok=True)

            fid_computation_size = 15000
            fid_computation_data = image_dataset["p_training"][torch.randperm(len(image_dataset["p_training"]))[:fid_computation_size]]

            with torch.no_grad():  # Don't track gradients for ODE solving
                if config.odeint_batch_size:
                    X_trajectory = batched_odeint(velocity_field, 
                                                  fid_computation_data,
                                                  timesteps,
                                                  config.odeint_batch_size,
                                                  ode_solver=config.ode_solver,
                                                  device=device)
                else:
                    X_trajectory = odeint(velocity_field, fid_computation_data.to(device), timesteps, method=config.ode_solver) 
                # (len(timesteps), training_size, dims, ...)

            fid = compute_fid_endpoint(vae_model, 
                                                image_dataset["q_rescale_factor"], 
                                                X_trajectory[-1,:,:,:,:],
                                                image_dataset["q_true_dataloader"], 
                                                config.odeint_batch_size, 
                                                device)
            
            print("FID at checkpoint {} : {}".format(training_dataloader.step+1, fid))

            # select 5 images for sanity check
            rand_idx_visualization = torch.randperm(len(p_batch))[:5]
            # (num_timesteps, len(rand_idx_visualization), channel, w, h)
            X_visualization_samples = X_trajectory[:,rand_idx_visualization,:,:,:].flatten(0,1)
            vae_model.eval()
            decoded_samples = autoencoder_decode(vae_model, 
                                                 X_visualization_samples, 
                                                 rescale_factor=image_dataset["q_rescale_factor"],
                                                 batch_size=None,
                                                 device=device)
            decoded_samples = decoded_samples.view((len(timesteps), 
                                                    len(rand_idx_visualization),
                                                    image_dataset["image_dim"][2],
                                                    image_dataset["image_dim"][0],
                                                    image_dataset["image_dim"][1])).cpu().detach()
            
            saving_file_name = os.path.join(ckpt_saving_dir, "5decoded_samples_trajectories.pdf")
            visualize_decoded_samples_trajectories(decoded_samples, 8, saving_file_name)

        training_pbar.update(1)

    torch.save(velocity_field.state_dict(), 
                os.path.join(config.saving_dir, 
                            'velocity_field_last.pt'))
    save_data(config.saving_dir + "loss_record.pkl", loss_record)