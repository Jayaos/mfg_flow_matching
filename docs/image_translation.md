

#### Image-to-image translation setup


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

Note that shoes, hangbags, and CelebA data should be located under `./data/` directory.

To reproduce the experiment, the trained VAE is required. VAEs should be trained following ["Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841).

For easier reproduce of the results, we provide the trained VAEs that can be downloaded downloaded from our [huggingface repository](https://huggingface.co/jayaos/vae_mfg_flow_matching):
```
git clone https://huggingface.co/jayaos/vae_mfg_flow_matching
```

You will see two VAE models with configuration files for each model:
- `vae_shoebags.ckpt`
- `vae_shoebags_config.yaml`
- `vae_celeba.ckpt`
- `vae_celeba_config.yaml`

We need to encode images into the latent space using the trained VAE. To encode the images using VAE, run ```encode_shoebags.py``` and ```encode_celeba.py```.

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

They were saved under `./data/` directory.