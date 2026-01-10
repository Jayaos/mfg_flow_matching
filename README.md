
# Experiment Guide


## High-dimensional Gaussian Benchmark

### 2-dimensional Gaussian mixture with visualization

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
                        saving_dir="/storage/home/hcoda1/0/jlee3541/p-yxie77-0/projects/mfg_flow/results/hd_gaussian/hdg_2_011026/")


run_mfg_flow_hdgaussian_2dvis(config, device=0)
```

```run_mfg_flow_hdgaussian_2dvis()``` has additional function to visualize the results.


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
                        saving_dir="/storage/home/hcoda1/0/jlee3541/p-yxie77-0/projects/mfg_flow/results/hd_gaussian/hdg_4_010726/")


run_mfg_flow_hdgaussian(config, device=0)
```