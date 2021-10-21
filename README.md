# Iterative Human and Automated Identification of Wildlife Images

[[Paper]](https://www.nature.com/articles/s42256-021-00393-0) [[Preprint]](https://arxiv.org/abs/2105.02320)

### *This repository is the author's re-implementation of the iterative recognition system with machine and humans in the loop described in:
"Iterative Human and Automated Identification of Wildlife Images"
Zhongqi Miao*,  Ziwei Liu*,  Kaitlyn M. Gaynor, Meredith S. Palmer, Stella X. Yu, Wayne M, Getz in Nature - Machine Intelligence, 2021*

<img src='./assets/intro.png' width=600>

Further information please contact [Zhongqi Miao](mailto:zhongqi.miao@berkeley.edu) and [Ziwei Liu](https://liuziwei7.github.io/).

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)
* [scikit-learn](https://scikit-learn.org/stable/)

## Data
All raw camera trap images that were used in this study (except classes with humans), along with the associated
annotation information, are uploaded to the publicly-available Labeled Information
Library of Alexandria: Biology and Conservation (LILA BC), and can be downloaded [[here]](https://lilablobssc.blob.core.windows.net/gorongosacameratraps/gorongosa-camera-traps-public-256x256.zip). 

## Changing dataset root for training and testing
Once the data is downloaded, please change the data root in the configuration files. For 
example: `dataset_root: /Mozambique`. 

## Stage 1: pre-training and evaluation
```
python main.py --config ./configs/Stage_1/plain_resnet_MOZ_S1_101920.yaml
```
## Stage 1: energy fine-tuning
```
python main.py --config ./configs/Stage_1/energy_resnet_MOZ_S1_101920.yaml --energy_ft
```
## Stage 2: training
```
python main.py --config ./configs/Stage_2/pslabel_oltr_resnet_MOZ_S2_111120.yaml
```
## Stage 2: energy fine-tuning
```
python main.py --config ./configs/Stage_2/pslabel_oltr_energy_resnet_MOZ_S2_111620.yaml
```
## Stage 2: deploying
```
python main.py --config ./configs/Stage_2/pslabel_oltr_energy_resnet_MOZ_S2_111620.yaml --deploy
```

## Demo
A demo of this code can be found in [[here]](https://codeocean.com/capsule/2011717/tree/v1) in CodeOcean.

## Citation
```
@article{10.1038/s42256-021-00393-0, 
year = {2021}, 
title = {{Iterative human and automated identification of wildlife images}}, 
author = {Miao, Zhongqi and Liu, Ziwei and Gaynor, Kaitlyn M and Palmer, Meredith S and Yu, Stella X and Getz, Wayne M}, 
journal = {Nature Machine Intelligence}, 
doi = {10.1038/s42256-021-00393-0}, 
pages = {885--895}, 
number = {10}, 
volume = {3}
}
```
