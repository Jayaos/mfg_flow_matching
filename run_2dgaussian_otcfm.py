from config import BaselineHDGaussianConfig
from src_run.run_hdgaussian import run_baselines_hdgaussian
import torch


config = BaselineHDGaussianConfig(dim=32,
                                  baseline="otcfm",
                                  velocity_field_hidden_dims=[64, 128, 256, 128, 64],
                                  velocity_field_layer_type="concatlinear",
                                  velocity_field_activation = torch.nn.ReLU(),
                                  learning_rate=0.001,
                                  training_batch_size=256,
                                  max_training_step=100,
                                  ode_solver="rk4",
                                  odeint_batch_size=2048,
                                  num_timesteps=10,
                                  checkpoint=50,
                                  saving_dir="C:/Users/jayao/Desktop/123/")


run_baselines_hdgaussian(config, device=0)