from config import MFGFlowHDGaussianConfig, BaselineHDGaussianConfig
from model import MLPClassifier, MLPVelocityField
from model import classifier_loss_fn, particle_optimization_loss_fn, vf_loss_fn
from utils import DataLoaderIterator, TrajectoryDataLoader
from utils import initialize_linear_interpolant, batched_odeint
from utils import load_data, save_data
from utils import compute_l2uvp_cos_forward
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


def run_mfg_flow_hdgaussian(config: MFGFlowHDGaussianConfig, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "mfg_flow_hdgaussian_config.pkl", config)

    timesteps = torch.linspace(0, 1, config.num_timesteps)
    timestep_size = 1/(config.num_timesteps-1)
    input_dim = config.dim
    benchmark = mbm.Mix3ToMix10Benchmark(input_dim)

    # initialize model
    print("data dimension: {}".format(input_dim))
    classifier = MLPClassifier(input_dim, config.classifier_hidden_dims, 
                               activation=config.classifier_activation).to(device)
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)

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
        elif config.velocity_field_initialization in ["linear-flow-matching"]:
            flow_matcher = LinearFlowMatcher()

        for i in tqdm(range(config.velocity_field_initialization_training_step)):
                
            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size)
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size)  
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

    # initial training of the classifier
    print("initial training of the classifier")
    for i in tqdm(range(config.classifier_initial_training_step)):

        p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size)
        q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size)

        with torch.no_grad():  # Don't track gradients for ODE solving
            if config.odeint_batch_size:
                p1_trajectory = batched_odeint(velocity_field, 
                                        p_batch, 
                                        timesteps,
                                        config.odeint_batch_size,
                                        ode_solver=config.ode_solver,
                                        device=device)
            else:
                p1_trajectory = odeint(velocity_field, p_batch, timesteps, method=config.ode_solver) 
        
        p1_batch = p1_trajectory[-1,:,:].to(device)
        q_batch = q_batch.to(device)
        classifier_loss = classifier_loss_fn(classifier, p1_batch, q_batch)
        classifier_optim.zero_grad()
        classifier_loss["loss"].backward()
        classifier_optim.step()
        initial_classifier_loss_record.append(classifier_loss["loss"].item())

    # Particle optimization
    kinetic_loss_record = []
    classifier_loss_record = []
    particle_optimization_loss_record = []
    intermediate_classifier_loss_record = []
    vf_loss_record = []
    L2_UVP_fwd_record = []
    cos_fwd_record = []

    for i in tqdm(range(config.epochs)):

        p_batch = benchmark.input_sampler.sample(config.epoch_data_size)
        q_batch = benchmark.output_sampler.sample(config.epoch_data_size)

        with torch.no_grad():  # Don't track gradients for ODE solving
            if config.odeint_batch_size:
                X_bar = batched_odeint(velocity_field, 
                                        p_batch, 
                                        timesteps,
                                        config.odeint_batch_size,
                                        ode_solver=config.ode_solver,
                                        device=device)
            else:
                X_bar = odeint(velocity_field, p_batch, timesteps, method=config.ode_solver) 

        # Prepare trajectory variables
        particle_0 = X_bar[0].clone().detach().unsqueeze(0).to(device) # (1, data_size, dim)
        particle_trajectory = X_bar[1:].clone().detach().to(device) # (num_timesteps - 1, data_size, dim)
        particle_trajectory.requires_grad_(True)
        particle_optim = torch.optim.Adam([particle_trajectory], lr=config.particle_optimization_learning_rate)

        particle_dataloader = TrajectoryDataLoader(particle_0, 
                                                    particle_trajectory,
                                                    batch_size=config.particle_optimization_batch_size, 
                                                    shuffle=True)
        
        for e in tqdm(range(config.particle_optimization_training_epoch)):

            for particle_0_batch, particle_trajectory_batch in particle_dataloader:

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

            # Update classifier every frequency
            if (e+1) % config.classifier_intermediate_training_frequency == 0:

                print("intermediate classifier update")
                p1_batch = particle_trajectory[-1, :, :].clone().detach() # (data_size, dim)
                intermediate_classifier_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p1_batch, q_batch), 
                                                                    batch_size=config.classifier_training_batch_size, 
                                                                    shuffle=True))
                for x_p, x_q in intermediate_classifier_dataloader:
                    x_p = x_p.to(device)
                    x_q = x_q.to(device)
                    if intermediate_classifier_dataloader.step <= config.classifier_intermediate_training_step:
                        classifier_loss = classifier_loss_fn(classifier, x_p, x_q)
                        classifier_optim.zero_grad()
                        classifier_loss["loss"].backward()
                        classifier_optim.step()
                        intermediate_classifier_loss_record.append(classifier_loss["loss"].item())
                    else:
                        break

        # updated particle trajectory
        # (len(timesteps), training_size, dim)
        X_bar = torch.cat([particle_0.detach().cpu(), particle_trajectory.detach().cpu()], dim=0) 
        X_bar = X_bar.transpose(1,0) # (training_size, len(timesteps), channel, h, w)
        vf_dataloader = DataLoaderIterator(DataLoader(TensorDataset(X_bar), 
                                            batch_size=config.velocity_field_training_batch_size, 
                                            shuffle=True))

        for x_traj_batch in vf_dataloader:
            
            # x_traj_batch is a tuple so x_traj_batch[0] is needed
            if vf_dataloader.step <= config.velocity_field_training_step:
                # (batch_size, len(timesteps), dim)
                vf_loss = vf_loss_fn(velocity_field, 
                                     x_traj_batch[0].to(device), 
                                     timesteps.to(device), 
                                     timestep_size)
                vf_optim.zero_grad()
                vf_loss["loss"].backward()
                vf_optim.step()
                vf_loss_record.append(vf_loss["loss"].detach().cpu().item())
            else:
                break

        if (i+1) % config.checkpoint == 0:

            L2_UVP_fwd, cos_fwd = compute_l2uvp_cos_forward(benchmark, 
                                                            velocity_field, 
                                                            config.num_timesteps, 
                                                            config.odeint_batch_size,
                                                            config.ode_solver, 
                                                            4096, 
                                                            device=device)
            
            L2_UVP_fwd_record.append(L2_UVP_fwd)
            cos_fwd_record.append(cos_fwd)
            print("$L^2$ UVP : {} at epoch {}".format(L2_UVP_fwd, i+1))
            print("cos similarity : {} at epoch {}".format(cos_fwd, i+1))

            loss_record[i] = {"kinetic_loss_record" : kinetic_loss_record,
                                "classifier_loss_record" : classifier_loss_record,
                                "intermediate_classifier_loss_record" : intermediate_classifier_loss_record,
                                "particle_optimization_loss_record" : particle_optimization_loss_record,
                                "vf_loss_record" : vf_loss_record,
                                "L2_UVP_fwd_record" : L2_UVP_fwd_record, 
                                "cos_fwd_record" : cos_fwd_record}
            
            save_data(config.saving_dir + "loss_record.pkl", loss_record)

    torch.save(classifier.state_dict(), os.path.join(config.saving_dir, 'classifier_e{}.pt'.format(i+1)))
    torch.save(velocity_field.state_dict(), os.path.join(config.saving_dir, 'velocity_field_e{}.pt'.format(i+1)))
    

def run_baselines_hdgaussian(config: BaselineHDGaussianConfig, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "baseline_config.pkl", config)

    input_dim = config.dim
    benchmark = mbm.Mix3ToMix10Benchmark(input_dim)

    # initialize model
    print("data dimension: {}".format(input_dim))
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)
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

        if i+1 <= config.max_training_step:
            
            p_batch = benchmark.input_sampler.sample(config.training_batch_size)
            q_batch = benchmark.output_sampler.sample(config.training_batch_size)
            p_batch.requires_grad_(False)
            q_batch.requires_grad_(False)
            t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, q_batch)
            t_batch = t_batch.to(device)
            xt_batch = xt_batch.to(device)
            ut_batch = ut_batch.to(device)
            vt_batch = velocity_field(t_batch, xt_batch)
            loss = torch.mean((vt_batch - ut_batch) ** 2)
            vf_optim.zero_grad()
            loss.backward()
            vf_optim.step()
            loss_list.append(loss.item())
        else:
            # config.max_training_step reached break the for loop
            break

        if ((i+1) % config.checkpoint) == 0:

            ckpt_saving_dir = os.path.join(config.saving_dir, "ckpt_{}".format(i+1))
            os.makedirs(ckpt_saving_dir, exist_ok=True)

            L2_UVP_fwd, cos_fwd = compute_l2uvp_cos_forward(benchmark, 
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
    

def run_baselines_hdgaussian_b(config: BaselineHDGaussianConfig, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "baseline_config.pkl", config)

    input_dim = config.dim
    benchmark = mbm.Mix3ToMix10Benchmark(input_dim)

    # initialize model
    print("data dimension: {}".format(input_dim))
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)
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

    p = benchmark.input_sampler.sample(100000)
    q = benchmark.output_sampler.sample(100000)
    vf_dataloader = DataLoaderIterator(DataLoader(TensorDataset(p, q), 
                                                  batch_size=config.training_batch_size, 
                                                  shuffle=True))
    vf_pbar = tqdm(total=config.max_training_step, desc="Velocity Field training")

    loss_record = dict()
    loss_list = []
    L2_UVP_fwd_list = []
    cos_fwd_list = []

    for p_batch, q_batch in vf_dataloader:

        if vf_dataloader.step <= config.max_training_step:

            t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, q_batch)
            t_batch = t_batch.to(device)
            xt_batch = xt_batch.to(device)
            ut_batch = ut_batch.to(device)
            vt_batch = velocity_field(t_batch, xt_batch)
            loss = torch.mean((vt_batch - ut_batch) ** 2)
            vf_optim.zero_grad()
            loss.backward()
            vf_optim.step()
            loss_list.append(loss.item())
            vf_pbar.update(1)
        else:
            # config.max_training_step reached break the for loop
            break

        if (vf_dataloader.step % config.checkpoint) == 0:

            ckpt_saving_dir = os.path.join(config.saving_dir, "ckpt_{}".format(vf_dataloader.step))
            os.makedirs(ckpt_saving_dir, exist_ok=True)

            L2_UVP_fwd, cos_fwd = compute_l2uvp_cos_forward(benchmark, 
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
            print("L2 UVP : {} at step {}".format(L2_UVP_fwd, vf_dataloader.step), flush=True)
            print("cos similarity : {} at step {}".format(cos_fwd, vf_dataloader.step), flush=True)
            save_data(config.saving_dir + "loss_record.pkl", loss_record)

    torch.save(velocity_field.state_dict(), os.path.join(config.saving_dir, 
                                                         'velocity_field_last.pt'))


def evaluate_baselines_hdgaussian(config_dir, velocity_field_dir, device):

    config = load_data(config_dir)

    input_dim = config.dim
    benchmark = mbm.Mix3ToMix10Benchmark(input_dim)

    # initialize model
    print("data dimension: {}".format(input_dim))
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)
    velocity_field.load_state_dict(torch.load(velocity_field_dir))

    L2_UVP_fwd, cos_fwd = compute_l2uvp_cos_forward(benchmark, 
                                                    velocity_field, 
                                                    config.num_timesteps, 
                                                    config.odeint_batch_size,
                                                    config.ode_solver, 
                                                    4096*4, 
                                                    device=device)
    print("L2 UVP : {}".format(L2_UVP_fwd))
    print("cos similarity : {}".format(cos_fwd))


def run_baselines_hdgaussian_exp(config: BaselineHDGaussianConfig, device):

    input_dim = config.dim
    benchmark = mbm.Mix3ToMix10Benchmark(input_dim)

    # initialize model
    print("data dimension: {}".format(input_dim))
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)
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

        if i+1 <= config.max_training_step:
            
            p_batch = torch.randn((config.training_batch_size, input_dim))
            q_batch = torch.randn((config.training_batch_size, input_dim))
            t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, q_batch)
            t_batch = t_batch.to(device)
            xt_batch = xt_batch.to(device)
            ut_batch = ut_batch.to(device)
            vt_batch = velocity_field(t_batch, xt_batch)
            loss = torch.mean((vt_batch - ut_batch) ** 2)
            vf_optim.zero_grad()
            loss.backward()
            vf_optim.step()
            loss_list.append(loss.item())
        else:
            # config.max_training_step reached break the for loop
            break