# High-dimensional Mean-Field Games by Particle-based Flow Matching

This repository contains code for the experiments in the paper **["High-dimensional Mean-Field Games by Particle-based Flow Matching"](https://arxiv.org/pdf/2512.01172)**.


## Setup

The current code was implemented on Python v3.9.21 and CUDA 12.8. Dependencies can be installed by requirements.txt.


## High-dimensional Gaussian Benchmark

The codes for the experiments on high-dimensional Gaussian benchmark are available in run_hd.py package. We used the same benchmark from **["Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark"](https://proceedings.neurips.cc/paper_files/paper/2021/file/7a6a6127ff85640ec69691fb0f7cb1a2-Paper.pdf)**, available at **[the authors' github](https://github.com/iamalexkorotin/Wasserstein2Benchmark)**.

Below is the implementation example on notebook.


### 2-dimensional Gaussian mixture with visualization

```run_mfg_flow_hdgaussian_2dvis()``` is for 2-dimensional Gaussian with visualization of the results.


```
from config import MFGFlowHDGaussianConfig
from run_hd import run_mfg_flow_hdgaussian, run_mfg_flow_hdgaussian_2dvis
import torch


config = MFGFlowHDGaussianConfig(dim=2,
                        classifier_hidden_dims=[2*4, 2*4, 2*4, 2*4],
                        classifier_activation=torch.nn.ReLU(),
                        classifier_learning_rate=0.001,
                        classifier_training_batch_size=1024,
                        classifier_initial_training_step=1000,
                        classifier_intermediate_training_frequency=5,
                        classifier_intermediate_training_step=1,
                        velocity_field_hidden_dims=[2*2, 2*4, 2*8, 2*4, 2*2],
                        velocity_field_layer_type="concatlinear",
                        velocity_field_activation = torch.nn.SiLU(),
                        velocity_field_learning_rate=0.001,
                        velocity_field_training_batch_size=1024,
                        velocity_field_initialization="linear-flow-matching",
                        velocity_field_initialization_training_step=10000,
                        velocity_field_training_step=1000,
                        particle_optimization_training_epoch=1000,
                        particle_optimization_learning_rate=0.001,
                        particle_optimization_batch_size=2048,
			kinetic_loss_weight=0.05,
			classifier_loss_weight=1.0,
                        epochs=200,
                        epoch_data_size=4096*5,
                        ode_solver="rk4",
                        odeint_batch_size=4096,
			num_timesteps=5,
                        saving_dir="")

run_mfg_flow_hdgaussian_2dvis(config, device=0)
```


### High-dimensional Gaussian mixture


```
from config import MFGFlowHDGaussianConfig
from run_hd import run_mfg_flow_hdgaussian
import torch


config = MFGFlowHDGaussianConfig(dim=4,
                       classifier_hidden_dims=[4*4, 4*4, 4*4, 4*4],
                       classifier_activation=torch.nn.ReLU(),
                        classifier_learning_rate=0.001,
                        classifier_training_batch_size=1024,
                        classifier_initial_training_step=2000,
                        classifier_intermediate_training_frequency=10,
                        classifier_intermediate_training_step=1,
                        velocity_field_hidden_dims=[4*2, 4*4, 4*8, 4*4, 4*2],
                        velocity_field_layer_type="concatlinear",
                        velocity_field_activation = torch.nn.ReLU(),
                        velocity_field_learning_rate=0.001,
                        velocity_field_training_batch_size=1024,
                        velocity_field_initialization="linear-flow-matching",
                        velocity_field_initialization_training_step=10000,
                        velocity_field_training_step=2000,
                        particle_optimization_training_epoch=2000,
                        particle_optimization_learning_rate=0.001,
                        particle_optimization_batch_size=2048,
			kinetic_loss_weight=0.05,
			classifier_loss_weight=1.0,
                        epochs=500,
                        epoch_data_size=4096*5,
                        ode_solver="rk4",
                        odeint_batch_size=4096,
			num_timesteps=5,
                        saving_dir="")


run_mfg_flow_hdgaussian(config, device=0)
```


## Citation

```bibtex
@article{yu2025high,
  title   = {High-dimensional Mean-Field Games by Particle-based Flow Matching},
  author  = {Yu, Jiajia and Lee, Junghwan and Xie, Yao and Cheng, Xiuyuan},
  journal = {arXiv preprint arXiv:2512.01172},
  year    = {2025}
}