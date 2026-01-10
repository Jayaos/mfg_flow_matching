from config import MFGFlowOTCelebAConfig, BaselineOTCelebAConfig
from model.classifier import ResNetClassifier
from model.velocity_field import ConvVelocityField
from model import classifier_loss_fn, particle_optimization_loss_fn, image_vf_loss_fn
from utils import DataLoaderIterator, TrajectoryDataLoader
from utils import batched_odeint
from utils import load_data, save_data
from utils.utils import compute_l2uvp_cos_forward_image, compute_l2uvp_cos_forward_particle_image, compute_l2uvp_cos_forward_vf_input_image
from utils.plotting import plot_random_otceleba_images
from baselines.torchcfm.conditional_flow_matching import VariancePreservingConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import LinearFlowMatcher
import baselines.wasserstein2bm.map_benchmark as mbm
from torchdiffeq import odeint
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import sys


def run_mfg_flow_otceleba(config: MFGFlowOTCelebAConfig, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "mfg_flow_otceleba_config.pkl", config)

    timesteps = torch.linspace(0, 1, config.num_timesteps)
    timestep_size = 1/(config.num_timesteps-1)
    input_dim = (3, 64, 64) # (channel, w, h)
    benchmark = mbm.CelebA64Benchmark(which=config.which_benchmark)

    # initialize model
    classifier = ResNetClassifier(input_dim[0],
                                  True,
                                  64,
                                  512,
                                  0.1).to(device)
    velocity_field = ConvVelocityField(input_dim[0], 
                                      config.velocity_field_encoding_dims,
                                      config.velocity_field_decoding_dims,
                                      config.velocity_field_kernel_sizes,
                                      config.velocity_field_strides).to(device)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.velocity_field_learning_rate)
    loss_record = dict()
    initial_classifier_loss_record = []

    if config.velocity_field_initialization:
        print("initialize velocity field using {}".format(config.velocity_field_initialization))

        if config.velocity_field_initialization in ["interflow", "stochastic-interpolant"]:
            flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
        elif config.velocity_field_initialization in ["conditional-flow-matching"]:
            flow_matcher = ConditionalFlowMatcher(sigma=0.1)
        elif config.velocity_field_initialization in ["otcfm", "ot-cfm"]:
            flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
        elif config.velocity_field_initialization in ["linear-flow-matching"]:
            flow_matcher = LinearFlowMatcher()

        for i in tqdm(range(config.velocity_field_initialization_training_step)):
                
            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size).reshape(-1, *input_dim).to("cpu")
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size).reshape(-1, *input_dim).to("cpu")
            t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, 
                                                                                            q_batch)
            t_batch = t_batch.to(device)
            xt_batch = xt_batch.to(device)
            ut_batch = ut_batch.to(device)
            vt_batch = velocity_field(t_batch, xt_batch)
            loss = torch.mean((vt_batch - ut_batch)**2)
            vf_optim.zero_grad()
            loss.backward()
            vf_optim.step()
    
    else:
        raise(ValueError, "velocity fields require initialization...")
    
    # evaluation after the initialization
    plot_random_otceleba_images(benchmark, 
                                velocity_field, 
                                config.num_timesteps, 
                                config.ode_solver, 
                                10, 
                                os.path.join(config.saving_dir, "sample_images_init.pdf"), 
                                device=device)
    L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward_image(benchmark, 
                                                             velocity_field, 
                                                             config.num_timesteps, 
                                                             config.odeint_batch_size,
                                                             config.ode_solver, 
                                                             4096, 
                                                             device=device)
    print("$L^2$ UVP : {} at initialization".format(L2_UVP_fwd))
    print("cos similarity : {} at initialization".format(cos_fwd))

    # initial classifier training
    p_training_epoch = benchmark.input_sampler.sample(config.epoch_training_size).reshape(-1, *input_dim).to("cpu")
    q_training_epoch = benchmark.output_sampler.sample(config.epoch_training_size).reshape(-1, *input_dim).to("cpu")

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

    # X_bar: (len(timesteps), data_size, dims, ...)
    p_1_training = X_bar[-1,:,:,:,:].detach() # (data_size, dim), endpoint of linear interpolant
    classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training_epoch), 
                                                            batch_size=config.classifier_training_batch_size, 
                                                            shuffle=True))

    for i, (x_p1, x_q) in enumerate(tqdm(classifier_dataloader)):

        x_p1 = x_p1.to(device)
        x_q = x_q.to(device)
        classifier_loss = classifier_loss_fn(classifier, x_p1, x_q)
        classifier_optim.zero_grad()
        classifier_loss["loss"].backward()
        classifier_optim.step()
        initial_classifier_loss_record.append(classifier_loss["loss"].item())

        if classifier_dataloader.step == config.classifier_initial_training_step:
            # config.initial_classifier_training_step reached break the for loop
            break

    for i in range(config.epochs):

        epoch_saving_dir = os.path.join(config.saving_dir, "epoch_{}".format(i+1))
        os.makedirs(epoch_saving_dir, exist_ok=True)

        # p_training and q_training have the same size
        # select small portion of training size for the epoch training
        p_training_epoch = benchmark.input_sampler.sample(config.epoch_training_size).reshape(-1, *input_dim).to("cpu")
        q_training_epoch = benchmark.output_sampler.sample(config.epoch_training_size).reshape(-1, *input_dim).to("cpu")

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
            
        del p_training_epoch # to save memory

        # Particle optimization
        ## before particle optimization, generate X_bar through initialization or solving ODE first
        ## and then batchify them, since batchfiy > solving ODE is more time consuming

        # Prepare trajectory variables
        particle_0 = X_bar[0].clone().detach().unsqueeze(0).to(device) # (1, data_size, 3, 64, 64)
        particle_trajectory = X_bar[1:].clone().detach().to(device) # (num_timesteps - 1, data_size, 3, 64, 64)
        particle_trajectory.requires_grad_(True)
        particle_optim = torch.optim.Adam([particle_trajectory], lr=config.particle_optimization_learning_rate)

        # evaluation of the particle trajectory before optimization
        L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward_particle_image(X_bar, benchmark, device=device)
        print("$L^2$ UVP : {} at epoch {} before particle optimization".format(L2_UVP_fwd, i+1))
        print("cos similarity : {} at epoch {} before particle optimization".format(cos_fwd, i+1))

        if i == 0:
            particle_optimization_epoch = config.initial_particle_optimization_epoch
        else:
            particle_optimization_epoch = config.particle_optimization_epoch

        kinetic_loss_record = []
        classifier_loss_record = []
        particle_optimization_loss_record = []

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

                p_1_training = particle_trajectory[-1, :, :, :, :].clone().detach() # (data_size, dims)
                classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p_1_training, q_training_epoch), 
                                                                    batch_size=config.classifier_training_batch_size, 
                                                                    shuffle=True))
                for x_p, x_q in classifier_dataloader:
                    x_p = x_p.to(device)
                    x_q = x_q.to(device)
                    if classifier_dataloader.step <= config.classifier_intermediate_training_step:
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

        # evaluation of the particle trajectory after optimization
        L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward_particle_image(X_bar, benchmark, device=device)
        print("$L^2$ UVP : {} at epoch {} after particle optimization".format(L2_UVP_fwd, i+1))
        print("cos similarity : {} at epoch {} after particle optimization".format(cos_fwd, i+1))

        # evaluation of the velocity field before training
        L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward_vf_input_image(X_bar[0,:,:], 
                                                                          benchmark, 
                                                                          velocity_field, 
                                                                          config.num_timesteps, 
                                                                          config.odeint_batch_size,
                                                                          config.ode_solver, 
                                                                          device=device)
        
        print("$L^2$ UVP : {} at epoch {} before velocity field training".format(L2_UVP_fwd, i+1))
        print("cos similarity : {} at epoch {} before velocity field training".format(cos_fwd, i+1))

        if i == 0:
            # at the first epoch, train velocity field more as warm-up
            velocity_field_training_step = config.initial_velocity_field_training_step
        else:
            velocity_field_training_step = config.velocity_field_training_step

        X_bar = X_bar.transpose(1,0) # (training_size, len(timesteps), channel, h, w)
        vf_dataloader = DataLoaderIterator(DataLoader(TensorDataset(X_bar), 
                                            batch_size=config.velocity_field_training_batch_size, 
                                            shuffle=True))
        vf_loss_record = []
        
        for x_traj_batch in range(velocity_field_training_step):

            x_traj_batch = next(vf_dataloader)
            # x_traj_batch is a tuple so x_traj_batch[0] is needed
            # (batch_size, len(timesteps), channel, w, h)
            vf_loss = image_vf_loss_fn(velocity_field, 
                                        x_traj_batch[0].to(device), 
                                        timesteps.to(device), 
                                        timestep_size)
            vf_optim.zero_grad()
            vf_loss["loss"].backward()
            vf_optim.step()
            vf_loss_record.append(vf_loss["loss"].detach().cpu().item())

            if vf_dataloader.step == velocity_field_training_step:
                break

        # evaluation of the velocity field after training
        L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward_vf_input_image(X_bar[:,0,:], 
                                                                          benchmark, 
                                                                          velocity_field, 
                                                                          config.num_timesteps, 
                                                                          config.odeint_batch_size,
                                                                          config.ode_solver, 
                                                                          device=device)
        print("$L^2$ UVP : {} at epoch {} after velocity field training".format(L2_UVP_fwd, i+1))
        print("cos similarity : {} at epoch {} after velocity field training".format(cos_fwd, i+1))

        # test evaluation at epoch i
        plot_random_otceleba_images(benchmark, 
                                    velocity_field, 
                                    config.num_timesteps, 
                                    config.ode_solver, 
                                    10, 
                                    os.path.join(config.saving_dir, "sample_images_e{}.pdf".format(i+1)), 
                                    device=device)
        L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward_image(benchmark, 
                                                                 velocity_field, 
                                                                 config.num_timesteps, 
                                                                 config.odeint_batch_size,
                                                                 config.ode_solver, 
                                                                 4096, 
                                                                 device=device)
        print("test $L^2$ UVP : {} at epoch {}".format(L2_UVP_fwd, i+1))
        print("test cos similarity : {} at epoch {}".format(cos_fwd, i+1))

        if i == 0:
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "initial_classifier_loss_record" : initial_classifier_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "intermediate_classifier_loss_record" : intermediate_classifier_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record}
        else:                
            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                              "classifier_loss_record" : classifier_loss_record,
                              "intermediate_classifier_loss_record" : intermediate_classifier_loss_record,
                              "particle_optimization_loss_record" : particle_optimization_loss_record,
                              "vf_loss_record" : vf_loss_record}
            
        save_data(config.saving_dir + "loss_record.pkl", loss_record)
        torch.save(classifier.state_dict(), os.path.join(epoch_saving_dir, 'classifier_e{}.pt'.format(i+1)))
        torch.save(velocity_field.state_dict(), os.path.join(epoch_saving_dir, 'velocity_field_e{}.pt'.format(i+1)))
    

def run_baselines_otceleba(config: BaselineOTCelebAConfig, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "baseline_config.pkl", config)

    input_dim = (3, 64, 64) # (channel, w, h)
    benchmark = mbm.CelebA64Benchmark(which=config.which_benchmark)

    # initialize model
    print("data dimension: {}".format(input_dim))
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

    loss_record = dict()
    loss_list = []
    L2_UVP_fwd_list = []
    cos_fwd_list = []

    for i in tqdm(range(config.max_training_step)):

        p_batch = benchmark.input_sampler.sample(config.training_batch_size).reshape(-1, *input_dim).to("cpu")
        q_batch = benchmark.output_sampler.sample(config.training_batch_size).reshape(-1, *input_dim).to("cpu")
        t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch.cpu(), q_batch.cpu())
        t_batch = t_batch.to(device)
        xt_batch = xt_batch.to(device)
        ut_batch = ut_batch.to(device)
        vt_batch = velocity_field(t_batch, xt_batch)
        loss = torch.mean((vt_batch - ut_batch) ** 2)
        vf_optim.zero_grad()
        loss.backward()
        vf_optim.step()
        loss_list.append(loss.item())

        if ((i+1) % config.checkpoint) == 0:

            #ckpt_saving_dir = os.path.join(config.saving_dir, "ckpt_{}".format(i+1))
            #os.makedirs(ckpt_saving_dir, exist_ok=True)

            plot_random_otceleba_images(benchmark, 
                                        velocity_field, 
                                        config.num_timesteps, 
                                        config.ode_solver, 
                                        10, 
                                        os.path.join(config.saving_dir, "sample_images_ckpt{}.pdf".format(i+1)), 
                                        device=device)
            
            L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward_image(benchmark, 
                                                                     velocity_field, 
                                                                     config.num_timesteps, 
                                                                     config.odeint_batch_size,
                                                                     config.ode_solver, 
                                                                     4096, 
                                                                     device=device)

            L2_UVP_fwd_list.append(L2_UVP_fwd)
            cos_fwd_list.append(cos_fwd)
            loss_record["L2_UVP_fwd_list"] = L2_UVP_fwd_list
            loss_record["cos_fwd_list"] = cos_fwd_list
            loss_record["loss_list"] = loss_list
            print("L2 UVP : {} at step {}".format(L2_UVP_fwd, i+1), flush=True)
            print("cos similarity : {} at step {}".format(cos_fwd, i+1), flush=True)
            save_data(config.saving_dir + "loss_record.pkl", loss_record)

    torch.save(velocity_field.state_dict(), os.path.join(config.saving_dir, 
                                                         'velocity_field_last.pt'))

