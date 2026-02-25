## High-dimensional Mean-Field Games by Particle-based Flow Matching

This repository contains code to reproduce the experiments in the paper **["High-dimensional Mean-Field Games by Particle-based Flow Matching"](https://arxiv.org/pdf/2512.01172)**.

### Environment Setup

Default setup was set to Python 3.9.21. Environment setup can be done as:

```
conda env create -f environment.yml

conda activate mfg_flow
```


### Toy Example

Running ```mfg_flow_toy_example.py``` will save models to the specified directory. The default hyperparameters setup will reproduct similar results in the paper. 

```
python mfg_flow_toy_example.py
```

Results will be saved in `./results/mfg_flow_toy_example/`. 

To plot trajectories, run ```mfg_flow_toy_example_plotting.py```:

```
python mfg_flow_toy_example_plotting.py
```

This will output particle trajectories before/after velocity field training at outer loop 1 and 10. Each plot will be saved at `./results/mfg_flow_toy_example/loop_1/particles_ode_trajectories_loop_1.pdf` and `./results/mfg_flow_toy_example/loop_10/particles_ode_trajectories_loop_10.pdf`


### Non-potential MFG

Running ```mfg_flow_nonpotential.py``` will obtain the results and plots in the paper.

```
python mfg_flow_nonpotential.py
```

Resulting plots will be saved under `./results/mfg_flow_nonpotential/`

### Image-to-image Translation

Details of setup for image-to-image translation is available [here](docs/image_translation.md).

#### Bags to shoes

Running ```mfg_flow_shoebags.py``` implements our method on image-to-image transltaion of bags to shoes. Default hyperparameters were set to the paper's setup. Details of the hyperparameters can be available in ```mfg_flow_shoebags.py```.

```
python mfg_flow_shoebags.py
```


#### CelebA male to female

Running ```mfg_flow_celeba.py``` implements our method on image-to-image transltaion of CelebA male to female. Default hyperparameters were set to the paper's setup. Details of the hyperparameters can be available in ```mfg_flow_celeba.py```.

```
python mfg_flow_celeba.py
```


## Citation

```bibtex
@article{yu2025high,
  title   = {High-dimensional Mean-Field Games by Particle-based Flow Matching},
  author  = {Yu, Jiajia and Lee, Junghwan and Xie, Yao and Cheng, Xiuyuan},
  journal = {arXiv preprint arXiv:2512.01172},
  year    = {2025}
}