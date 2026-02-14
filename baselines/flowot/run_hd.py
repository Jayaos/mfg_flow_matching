from baselines.flowot.config import FlowOTHDGaussianConfig
from baselines.flowot.model.classifier import MLPClassifier
from baselines.flowot.model.velocity_field import MLPVelocityField
from baselines.flowot.utils import load_data, save_data
from baselines.flowot.utils import plot_losses, batched_odeint
from baselines.flowot.model.loss import compute_logit_loss, compute_2wasserstein_loss
from baselines.torchcfm.conditional_flow_matching import VariancePreservingConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from baselines.torchcfm.conditional_flow_matching import LinearFlowMatcher
import baselines.wasserstein2bm.map_benchmark as mbm
from utils.utils import compute_l2uvp_cos_forward
from utils.plotting import plot_2d_gaussian_samples
from torchdiffeq import odeint
import torch
from tqdm import tqdm
import os


def run_flowot_hdgaussian(config: FlowOTHDGaussianConfig, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "flowot_config.pkl", config)

    timesteps = torch.linspace(0, 1, config.num_timesteps)
    timesteps_reversed = torch.linspace(1, 0, config.num_timesteps)
    input_dim = config.dim
    benchmark = mbm.Mix3ToMix10Benchmark(input_dim)

    # initialize model
    print("data dimension: {}".format(input_dim))
    classifier_pq = MLPClassifier(input_dim, 
                                  config.classifier_hidden_dims, 
                                  activation=config.classifier_activation).to(device)
    classifier_qp = MLPClassifier(input_dim, 
                                  config.classifier_hidden_dims, 
                                  activation=config.classifier_activation).to(device)
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)
    
    # optimizer
    classifier_pq_optim = torch.optim.Adam(classifier_pq.parameters(), lr=config.classifier_learning_rate)
    classifier_qp_optim = torch.optim.Adam(classifier_qp.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.velocity_field_learning_rate)

    # initialization of the velocity field
    if config.velocity_field_initialization:

        print("initialize velocity field using {}".format(config.velocity_field_initialization))

        if config.velocity_field_initialization in ["interflow", "stochastic-interpolant"]:
            flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
        elif config.velocity_field_initialization in ["conditional-flow-matching"]:
            flow_matcher = ConditionalFlowMatcher(sigma=0.1)
        elif config.velocity_field_initialization in ["linear-flow-matching"]:
            flow_matcher = LinearFlowMatcher()

        for i in tqdm(range(config.velocity_field_initialization_training_step)):
                
            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, q_batch)
            t_batch = t_batch.to(device)
            xt_batch = xt_batch.to(device)
            ut_batch = ut_batch.to(device)
            vt_batch = velocity_field(t_batch, xt_batch)
            loss = torch.mean((vt_batch - ut_batch)**2)
            vf_optim.zero_grad()
            loss.backward()
            vf_optim.step()

    # evaluation of the initial velocity field
    L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward(benchmark, 
                                                       velocity_field, 
                                                       config.num_timesteps, 
                                                       config.odeint_batch_size,
                                                       config.ode_solver, 
                                                       4096*4, 
                                                       device=device)
    
    print("$L^2$ UVP : {} at initialization".format(L2_UVP_fwd))
    print("cos similarity : {} at initialization".format(cos_fwd))

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
            for i in tqdm(range(config.classifier_initial_training_step)):
                p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

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
                classifier_pq_loss = compute_logit_loss(classifier_pq, p1_batch, q_batch)
                classifier_pq_optim.zero_grad()
                classifier_pq_loss.backward()
                classifier_pq_optim.step()
                initial_pq_classifier_loss_record.append(classifier_pq_loss.item())

        # training velocity field in forward direction
        print("training velocity field in the forward direction...")
        ckpt_step = 0 # step count for checkpoint
        for i in tqdm(range(config.velocity_field_training_step)):
            ckpt_step += 1

            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size).to("cpu")

            # need gradient for solving ODE here
            if config.odeint_batch_size:
                x_p_trajectory_batch = batched_odeint(velocity_field, 
                                                      p_batch, 
                                                      timesteps,
                                                      config.odeint_batch_size,
                                                      ode_solver=config.ode_solver,
                                                      device=device)
            else:
                x_p_trajectory_batch = odeint(velocity_field, p_batch, timesteps, method=config.ode_solver) 

            # permute to (batch_size, len(timesteps), dims, ...)
            x_p_trajectory_batch = x_p_trajectory_batch.permute(1,0,2).to(device)
            kl_loss_pq = -classifier_pq(x_p_trajectory_batch[:,-1,:]).mean()
            wasserstein_loss_pq = compute_2wasserstein_loss(x_p_trajectory_batch, 
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
                for j in tqdm(range(config.classifier_intermediate_training_step)):
                    
                    p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                    q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

                    with torch.no_grad():  # Don't track gradients for ODE solving
                        if config.odeint_batch_size:
                            x_p_trajectory_batch = batched_odeint(velocity_field, 
                                                                  p_batch, 
                                                                  timesteps,
                                                                  config.odeint_batch_size,
                                                                  ode_solver=config.ode_solver,
                                                                  device=device)
                        else:
                            x_p_trajectory_batch = odeint(velocity_field, 
                                                          p_batch, 
                                                          timesteps, 
                                                          method=config.ode_solver) 
                                
                    x_p1_batch = x_p_trajectory_batch[-1,:,:].to(device) # (sample_size, dims)
                    q_batch = q_batch.to(device)
                    classifier_pq_loss = compute_logit_loss(classifier_pq, x_p1_batch, q_batch)
                    classifier_pq_optim.zero_grad()
                    classifier_pq_loss.backward()
                    classifier_pq_optim.step()
                    pq_intermediate_classifier_loss_record.append(classifier_pq_loss.item())

            if (ckpt_step+1) % config.checkpoint_step == 0:

                plot_losses(kl_loss_pq_record, 
                            wasserstein_loss_pq_record, 
                            pq_refinement_loss_record,
                            config.saving_dir,
                            "loss_PQ_log_e{}_step{}.pdf".format(e+1, ckpt_step+1))
                
                # evaluation of the velocity field
                L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward(benchmark, 
                                                                   velocity_field, 
                                                                   config.num_timesteps, 
                                                                   config.odeint_batch_size,
                                                                   config.ode_solver, 
                                                                   4096*4, 
                                                                   device=device)
                
                print("$L^2$ UVP at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, L2_UVP_fwd))
                print("cos similarity at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, cos_fwd))
                             
        # training classifier_qp at the first epoch
        if e == 0:

            print("training classifier qp at the first epoch...")
            for i in tqdm(range(config.classifier_initial_training_step)):
                p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

                with torch.no_grad():  # Don't track gradients for ODE solving

                    if config.odeint_batch_size:
                        q0_trajectory = batched_odeint(velocity_field, 
                                                       q_batch, 
                                                       timesteps_reversed,
                                                       config.odeint_batch_size,
                                                       ode_solver=config.ode_solver,
                                                       device=device)
                    else:
                        q0_trajectory = odeint(velocity_field, q_batch, timesteps_reversed, method=config.ode_solver)

                q0_batch = q0_trajectory[-1,:,:].to(device)
                p_batch = p_batch.to(device)
                classifier_qp_loss = compute_logit_loss(classifier_qp, q0_batch, p_batch)
                classifier_qp_optim.zero_grad()
                classifier_qp_loss.backward()
                classifier_qp_optim.step()
                initial_qp_classifier_loss_record.append(classifier_qp_loss.item())

        # training velocity field in backward direction
        print("training velocity field in the backward direction...")
        ckpt_step = 0
        for i in tqdm(range(config.velocity_field_training_step)):
            ckpt_step += 1

            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size).to("cpu")

            # need gradient for solving ODE here
            if config.odeint_batch_size:
                x_q_trajectory_batch = batched_odeint(velocity_field, 
                                                      q_batch, 
                                                      timesteps_reversed,
                                                      config.odeint_batch_size,
                                                      ode_solver=config.ode_solver,
                                                      device=device)
            else:
                x_q_trajectory_batch = odeint(velocity_field, q_batch, timesteps_reversed, method=config.ode_solver) 

            # permute to (batch_size, len(timesteps), dims, ...)
            x_q_trajectory_batch = x_q_trajectory_batch.permute(1,0,2).to(device)
            kl_loss_qp = -classifier_qp(x_q_trajectory_batch[:,-1,:]).mean()
            wasserstein_loss_qp = compute_2wasserstein_loss(x_q_trajectory_batch, 
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

                for j in tqdm(range(config.classifier_intermediate_training_step)):
                    
                    p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                    q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

                    with torch.no_grad():  # Don't track gradients for ODE solving
                        if config.odeint_batch_size:
                            x_q_trajectory_batch = batched_odeint(velocity_field, 
                                                                  q_batch, 
                                                                  timesteps_reversed,
                                                                  config.odeint_batch_size,
                                                                  ode_solver=config.ode_solver,
                                                                  device=device)
                        else:
                            x_q_trajectory_batch = odeint(velocity_field, 
                                                          q_batch, 
                                                          timesteps_reversed, 
                                                          method=config.ode_solver) 
                                
                    q0_batch = x_q_trajectory_batch[-1,:,:].to(device) # (sample_size, dims)
                    p_batch = p_batch.to(device)
                    classifier_qp_loss = compute_logit_loss(classifier_qp, q0_batch, p_batch)
                    classifier_qp_optim.zero_grad()
                    classifier_qp_loss.backward()
                    classifier_qp_optim.step()
                    qp_intermediate_classifier_loss_record.append(classifier_qp_loss.item())

            if (ckpt_step+1) % config.checkpoint_step == 0:

                plot_losses(kl_loss_qp_record, 
                            wasserstein_loss_qp_record, 
                            qp_refinement_loss_record,
                            config.saving_dir,
                            "loss_QP_log_e{}_step{}.pdf".format(e+1, ckpt_step+1))
                
                # evaluation of the velocity field
                L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward(benchmark, 
                                                                    velocity_field, 
                                                                    config.num_timesteps, 
                                                                    config.odeint_batch_size,
                                                                    config.ode_solver, 
                                                                    4096*4, 
                                                                    device=device)
                
                print("$L^2$ UVP at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, L2_UVP_fwd))
                print("cos similarity at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, cos_fwd))
                
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


def run_flowot_hdgaussian_2dvisualization(config: FlowOTHDGaussianConfig, device):

    os.makedirs(config.saving_dir, exist_ok=True)
    save_data(config.saving_dir + "flowot_config.pkl", config)

    timesteps = torch.linspace(0, 1, config.num_timesteps)
    timesteps_reversed = torch.linspace(1, 0, config.num_timesteps)
    input_dim = config.dim
    benchmark = mbm.Mix3ToMix10Benchmark(input_dim)

    # initialize model
    print("data dimension: {}".format(input_dim))
    classifier_pq = MLPClassifier(input_dim, 
                                  config.classifier_hidden_dims, 
                                  activation=config.classifier_activation).to(device)
    classifier_qp = MLPClassifier(input_dim, 
                                  config.classifier_hidden_dims, 
                                  activation=config.classifier_activation).to(device)
    velocity_field = MLPVelocityField(input_dim, 
                                      1, # time dim = 1
                                      config.velocity_field_hidden_dims, 
                                      config.velocity_field_layer_type,
                                      config.velocity_field_activation).to(device)
    
    # optimizer
    classifier_pq_optim = torch.optim.Adam(classifier_pq.parameters(), lr=config.classifier_learning_rate)
    classifier_qp_optim = torch.optim.Adam(classifier_qp.parameters(), lr=config.classifier_learning_rate)
    vf_optim = torch.optim.Adam(velocity_field.parameters(), lr=config.velocity_field_learning_rate)

    # initialization of the velocity field
    if config.velocity_field_initialization:

        print("initialize velocity field using {}".format(config.velocity_field_initialization))

        if config.velocity_field_initialization in ["interflow", "stochastic-interpolant"]:
            flow_matcher = VariancePreservingConditionalFlowMatcher(sigma=0.1)
        elif config.velocity_field_initialization in ["conditional-flow-matching"]:
            flow_matcher = ConditionalFlowMatcher(sigma=0.1)
        elif config.velocity_field_initialization in ["linear-flow-matching"]:
            flow_matcher = LinearFlowMatcher()

        for i in tqdm(range(config.velocity_field_initialization_training_step)):
                
            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            t_batch, xt_batch, ut_batch = flow_matcher.sample_location_and_conditional_flow(p_batch, q_batch)
            t_batch = t_batch.to(device)
            xt_batch = xt_batch.to(device)
            ut_batch = ut_batch.to(device)
            vt_batch = velocity_field(t_batch, xt_batch)
            loss = torch.mean((vt_batch - ut_batch)**2)
            vf_optim.zero_grad()
            loss.backward()
            vf_optim.step()

    # evaluation of the initial velocity field
    L2_UVP_fwd, cos_fwd, samples_result = compute_l2uvp_cos_forward(benchmark, 
                                                       velocity_field, 
                                                       config.num_timesteps, 
                                                       config.odeint_batch_size,
                                                       config.ode_solver, 
                                                       4096*4, 
                                                       device=device)
    
    plot_2d_gaussian_samples(samples_result, 
                             os.path.join(config.saving_dir, "sanity_check_plot_init.pdf"))
    
    print("$L^2$ UVP : {} at initialization".format(L2_UVP_fwd))
    print("cos similarity : {} at initialization".format(cos_fwd))

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
            for i in tqdm(range(config.classifier_initial_training_step)):
                p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

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
                classifier_pq_loss = compute_logit_loss(classifier_pq, p1_batch, q_batch)
                classifier_pq_optim.zero_grad()
                classifier_pq_loss.backward()
                classifier_pq_optim.step()
                initial_pq_classifier_loss_record.append(classifier_pq_loss.item())

        # training velocity field in forward direction
        print("training velocity field in the forward direction...")
        ckpt_step = 0 # step count for checkpoint
        for i in tqdm(range(config.velocity_field_training_step)):
            ckpt_step += 1

            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size).to("cpu")

            # need gradient for solving ODE here
            if config.odeint_batch_size:
                x_p_trajectory_batch = batched_odeint(velocity_field, 
                                                      p_batch, 
                                                      timesteps,
                                                      config.odeint_batch_size,
                                                      ode_solver=config.ode_solver,
                                                      device=device)
            else:
                x_p_trajectory_batch = odeint(velocity_field, p_batch, timesteps, method=config.ode_solver) 

            # permute to (batch_size, len(timesteps), dims, ...)
            x_p_trajectory_batch = x_p_trajectory_batch.permute(1,0,2).to(device)
            kl_loss_pq = -classifier_pq(x_p_trajectory_batch[:,-1,:]).mean()
            wasserstein_loss_pq = compute_2wasserstein_loss(x_p_trajectory_batch, 
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
                for j in tqdm(range(config.classifier_intermediate_training_step)):
                    
                    p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                    q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

                    with torch.no_grad():  # Don't track gradients for ODE solving
                        if config.odeint_batch_size:
                            x_p_trajectory_batch = batched_odeint(velocity_field, 
                                                                  p_batch, 
                                                                  timesteps,
                                                                  config.odeint_batch_size,
                                                                  ode_solver=config.ode_solver,
                                                                  device=device)
                        else:
                            x_p_trajectory_batch = odeint(velocity_field, 
                                                          p_batch, 
                                                          timesteps, 
                                                          method=config.ode_solver) 
                                

                    x_p1_batch = x_p_trajectory_batch[-1,:,:].to(device) # (sample_size, dims)
                    q_batch = q_batch.to(device)
                    classifier_pq_loss = compute_logit_loss(classifier_pq, x_p1_batch, q_batch)
                    classifier_pq_optim.zero_grad()
                    classifier_pq_loss.backward()
                    classifier_pq_optim.step()
                    pq_intermediate_classifier_loss_record.append(classifier_pq_loss.item())

            if (ckpt_step+1) % config.checkpoint_step == 0:

                plot_losses(kl_loss_pq_record, 
                            wasserstein_loss_pq_record, 
                            pq_refinement_loss_record,
                            config.saving_dir,
                            "loss_PQ_log_e{}_step{}.pdf".format(e+1, ckpt_step+1))
                
                # evaluation of the velocity field
                L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward(benchmark, 
                                                                   velocity_field, 
                                                                   config.num_timesteps, 
                                                                   config.odeint_batch_size,
                                                                   config.ode_solver, 
                                                                   4096*4, 
                                                                   device=device)
                plot_2d_gaussian_samples(samples_result, 
                                         os.path.join(config.saving_dir, 
                                                      "sanity_check_plot_ckpt{}_forward.pdf".format(ckpt_step+1)))
                
                print("$L^2$ UVP at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, L2_UVP_fwd))
                print("cos similarity at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, cos_fwd))
                             
        # training classifier_qp at the first epoch
        if e == 0:

            print("training classifier qp at the first epoch...")
            for i in tqdm(range(config.classifier_initial_training_step)):
                p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

                with torch.no_grad():  # Don't track gradients for ODE solving

                    if config.odeint_batch_size:
                        q0_trajectory = batched_odeint(velocity_field, 
                                                       q_batch, 
                                                       timesteps_reversed,
                                                       config.odeint_batch_size,
                                                       ode_solver=config.ode_solver,
                                                       device=device)
                    else:
                        q0_trajectory = odeint(velocity_field, q_batch, timesteps_reversed, method=config.ode_solver)

                q0_batch = q0_trajectory[-1,:,:].to(device)
                p_batch = p_batch.to(device)
                classifier_qp_loss = compute_logit_loss(classifier_qp, q0_batch, p_batch)
                classifier_qp_optim.zero_grad()
                classifier_qp_loss.backward()
                classifier_qp_optim.step()
                initial_qp_classifier_loss_record.append(classifier_qp_loss.item())

        # training velocity field in backward direction
        print("training velocity field in the backward direction...")
        ckpt_step = 0
        for i in tqdm(range(config.velocity_field_training_step)):
            ckpt_step += 1

            p_batch = benchmark.input_sampler.sample(config.velocity_field_training_batch_size).to("cpu")
            q_batch = benchmark.output_sampler.sample(config.velocity_field_training_batch_size).to("cpu")

            # need gradient for solving ODE here
            if config.odeint_batch_size:
                x_q_trajectory_batch = batched_odeint(velocity_field, 
                                                      q_batch, 
                                                      timesteps_reversed,
                                                      config.odeint_batch_size,
                                                      ode_solver=config.ode_solver,
                                                      device=device)
            else:
                x_q_trajectory_batch = odeint(velocity_field, q_batch, timesteps_reversed, method=config.ode_solver) 

            # permute to (batch_size, len(timesteps), dims, ...)
            x_q_trajectory_batch = x_q_trajectory_batch.permute(1,0,2).to(device)
            kl_loss_qp = -classifier_qp(x_q_trajectory_batch[:,-1,:]).mean()
            wasserstein_loss_qp = compute_2wasserstein_loss(x_q_trajectory_batch, 
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
                for j in tqdm(range(config.classifier_intermediate_training_step)):
                    
                    p_batch = benchmark.input_sampler.sample(config.classifier_training_batch_size).to("cpu")
                    q_batch = benchmark.output_sampler.sample(config.classifier_training_batch_size).to("cpu")

                    with torch.no_grad():  # Don't track gradients for ODE solving
                        if config.odeint_batch_size:
                            x_q_trajectory_batch = batched_odeint(velocity_field, 
                                                                  q_batch, 
                                                                  timesteps_reversed,
                                                                  config.odeint_batch_size,
                                                                  ode_solver=config.ode_solver,
                                                                  device=device)
                        else:
                            x_q_trajectory_batch = odeint(velocity_field, 
                                                          q_batch, 
                                                          timesteps_reversed, 
                                                          method=config.ode_solver)
                                
                    q0_batch = x_q_trajectory_batch[-1,:,:].to(device) # (sample_size, dims)
                    p_batch = p_batch.to(device)
                    classifier_qp_loss = compute_logit_loss(classifier_qp, q0_batch, p_batch)
                    classifier_qp_optim.zero_grad()
                    classifier_qp_loss.backward()
                    classifier_qp_optim.step()
                    qp_intermediate_classifier_loss_record.append(classifier_qp_loss.item())

            if (ckpt_step+1) % config.checkpoint_step == 0:

                plot_losses(kl_loss_qp_record, 
                            wasserstein_loss_qp_record, 
                            qp_refinement_loss_record,
                            config.saving_dir,
                            "loss_QP_log_e{}_step{}.pdf".format(e+1, ckpt_step+1))
                
                # evaluation of the velocity field
                L2_UVP_fwd, cos_fwd, _ = compute_l2uvp_cos_forward(benchmark, 
                                                                    velocity_field, 
                                                                    config.num_timesteps, 
                                                                    config.odeint_batch_size,
                                                                    config.ode_solver, 
                                                                    4096*4, 
                                                                    device=device)
                
                plot_2d_gaussian_samples(samples_result, 
                                         os.path.join(config.saving_dir, 
                                                      "sanity_check_plot_ckpt{}_backward.pdf".format(ckpt_step+1)))
                
                print("$L^2$ UVP at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, L2_UVP_fwd))
                print("cos similarity at ckpt {}, forward iter {}: {}".format(ckpt_step+1, e, cos_fwd))
                
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
