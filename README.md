## High-dimensional Mean-Field Games by Particle-based Flow Matching

This repository contains code to reproduce the experiments in the paper **["High-dimensional Mean-Field Games by Particle-based Flow Matching"](https://arxiv.org/pdf/2512.01172)**.

### Environment Setup

Default setup was set to Python 3.9.21. Environment setup can be done as:

```
conda env create -f environment.yml

conda activate mfg_flow
```


### Toy Example

```
python mfg_flow_toy_example.py
```


### Image-to-image Translation

#### Shoes to bags

To download the data:
```
wget http://efrosgans.eecs.berkeley.edu/iGAN/datasets/shoes_64.zip
wget http://efrosgans.eecs.berkeley.edu/iGAN/datasets/handbag_64.zip
```
and unzip those files to obtain

```
shoes_64.hdf5
handbag_64.hdf5
```


#### CelebA

CelebA data can be downloaded at [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).



## Citation

```bibtex
@article{yu2025high,
  title   = {High-dimensional Mean-Field Games by Particle-based Flow Matching},
  author  = {Yu, Jiajia and Lee, Junghwan and Xie, Yao and Cheng, Xiuyuan},
  journal = {arXiv preprint arXiv:2512.01172},
  year    = {2025}
}