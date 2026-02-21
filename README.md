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
python mfg_flow_toy_example.py --saving_dir 'directory to save'
```

To plot trajectories, use ```mfg_flow_toy_example_plotting.py``` as:

```
python mfg_flow_toy_example_plotting.py --model_dir '' --config_dir '' --particle_trajectories_dir '' --saving_dir ''
```


### Non-potential MFG

Running ```mfg_flow_nonpotential.py``` will obtain the results and plots in the paper.

```
python mfg_flow_nonpotential.py --save-plots 'directory to save plots'
```

### Image-to-image Translation

Images of shoes and handbags can be downloaded from [here](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/).

Or directly via the command line:
```
wget http://efrosgans.eecs.berkeley.edu/iGAN/datasets/shoes_64.zip
wget http://efrosgans.eecs.berkeley.edu/iGAN/datasets/handbag_64.zip
```

After unzipping, you should obtain the following files:
- `shoes_64.hdf5`
- `handbag_64.hdf5`

CelebA data can be downloaded from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

To reproduce the experiment, the trained VAE also need to be downloaded from our [huggingface repository](https://huggingface.co/jayaos/vae_mfg_flow_matching):
```
git clone https://huggingface.co/jayaos/vae_mfg_flow_matching
```

You will see two VAE models with configuration files:
- `vae_shoebags.ckpt`
- `vae_shoebags_config.yaml`
- `vae_celeba.ckpt`
- `vae_celeba_config.yaml`

We need to encode images into the latent space using the trained VAE. To encode the images using VAE, run ```encode_shoebags.py``` and ```encode_celeba.py```.

Or download from our [huggingface dataset repository](https://huggingface.co/datasets/jayaos/vae_encoded_mfg_flow_matching):
```
git clone https://huggingface.co/datasets/jayaos/vae_encoded_mfg_flow_matching
```

This will give encoded train and test images for image-to-image translation on bags to shoes:
- `train_encoded_bags.pt`
- `train_encoded_shoes.pt`
- `test_encoded_bags.pt`
- `test_encoded_shoes.pt`

and for CelebA male to female.
- `train_encoded_male.pt`
- `train_encoded_female.pt`
- `test_encoded_male.pt`
- `test_encoded_female.pt`


#### Bags to shoes

Running ```mfg_flow_shoebags.py``` implements our method on image-to-image transltaion of bags to shoes. Default hyperparameters were set to the paper's setup. Directories of the trained VAE, VAE config, raw data, and encoded data need to be specified as below. Details of the hyperparameters can be available in ```mfg_flow_shoebags.py```.

```
python mfg_flow_shoebags.py --vae_config_dir ./vae_mfg_flow_matching/vae_shoebags_config.yaml --vae_model_dir ./vae_mfg_flow_matching/vae_shoebags.ckpt --shoes_dataset_dir ./shoes_64.hdf5 --bags_dataset_dir ./handbag_64.hdf5 --train_encoded_shoes_dir ./train_encoded_shoes.pt --train_encoded_bags_dir ./train_encoded_bags.pt --test_encoded_shoes_dir ./test_encoded_shoes.pt --test_encoded_bags_dir ./test_encoded_bags.pt
```



#### CelebA male to female

Running ```mfg_flow_celeba.py``` implements our method on image-to-image transltaion of CelebA male to female. Default hyperparameters were set to the paper's setup. Directories of the trained VAE, VAE config, raw data, and encoded data need to be specified as below. Details of the hyperparameters can be available in ```mfg_flow_celeba.py```.

```
python mfg_flow_celeba.py --vae_config_dir ./vae_mfg_flow_matching/vae_celeba_config.yaml --vae_model_dir ./vae_mfg_flow_matching/vae_celeba.ckpt --data_dir ./celeba/ --train_encoded_male_dir ./train_encoded_male.pt --train_encoded_female_dir ./train_encoded_female.pt --test_encoded_male_dir ./test_encoded_male.pt --test_encoded_female_dir ./test_encoded_female.pt
```


## Citation

```bibtex
@article{yu2025high,
  title   = {High-dimensional Mean-Field Games by Particle-based Flow Matching},
  author  = {Yu, Jiajia and Lee, Junghwan and Xie, Yao and Cheng, Xiuyuan},
  journal = {arXiv preprint arXiv:2512.01172},
  year    = {2025}
}