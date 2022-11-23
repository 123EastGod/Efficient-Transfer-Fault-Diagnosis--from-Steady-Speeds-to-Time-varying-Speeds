# An Efficient Transfer Fault Diagnosis  from Steady Speeds to Time-varying Speeds

PyTorch codes for paper "Unsupervised domain-share CNN for machine fault transfer diagnosis from steady speeds to time-varying speeds ".

# Experimental Datasets

1.Rotor-bearing system simulation model datasets 

Introduction to datasets:

| Rotating speed | Number of samples | Number of points in each sample |
| -------------- | ----------------- | ------------------------------- |
| 1909 rpm       | 4 * 500( 2000)    | 1024                            |

More details of the test rig can be found : 

Yu K, Fu Q, Ma H, Lin T, Li X. Simulation data driven weakly supervised adversarial domain adaptation approach for intelligent cross-machine fault diagnosis. Struct Health Monit 2021:1475921720980718. 

2.Roller bearings datasets working under speed fluctuation conditions

Introduction to datasets:

| Rotating speed                                   | Number of samples | Number of points in each sample                 |
| ------------------------------------------------ | ----------------- | ----------------------------------------------- |
| Time-varying speeds (640 rpm -1500 rpm -640 rpm) | 4 * 500 (2000)    | 1024More details of the test rig can be found : |

Wang J, Li S, Han B, An Z, Xin Y, Qian W, et al. Construction of a batch-normalized autoencoder network and its application in mechanical intelligent fault diagnosis. Meas Sci Technol 2018;30(1):015106. 

3.Fault simulation test bed fault datasets collect from HNU

Details about Bearing fault specifications:

| Fault location | Diameter | Depth  | Bearing manufacturer |
| -------------- | -------- | ------ | -------------------- |
| Inner raceway  | 0.15mm   | 0.13mm | SKF 6307             |
| Outer raceway  | 0.15mm   | 0.13mm | SKF 6307             |
| Ball           | 0.15mm   | 0.10mm | SKF 6307             |

Descriptions about the eight health states:

| Health states of gear | Health states of bearing |
| --------------------- | ------------------------ |
| Breakage              | Good                     |
| Breakage              | Roller fault             |
| Breakage              | Inner race fault         |
| Breakage              | Outer race fault         |
| Crack                 | Good                     |
| Crack                 | Roller fault             |
| Crack                 | Inner race fault         |
| Crack                 | Outer race fault         |

Introduction to datasets:

| Rotating speed                                            | Number of samples | Number of points in each sample |
| --------------------------------------------------------- | ----------------- | ------------------------------- |
| 600 rpm & Time-varying speeds (300 rpm -800 rpm -300 rpm) | 8 *200 (1600)     | 1024                            |

# Programming Environment

All experiments are performed on a computer with single GeForce GTX 1650Ti, Intel Core i5-10700F running under Python 3.7 and Pytorch 1.2.

- Python 3.7
- Numpy 1.16
- Pandas 0.24
- sklearn 0.21
- Scipy 1.2.1
- pytorch 1.2
- torchvision >= 0.40

## How to Use

For example, use the following commands to test MK-MMD for HNU with the transfer_task 2 ->1

- `python train_advanced.py --data_name HNU --data_dir D:/Data/HNU --transfer_task [2],[1] --last_batch True --distance_metric True --distance_loss MK-MMD `

## Acknowledgement

The code is designed based on [UDTL](https://github.com/ZhaoZhibin/UDTL).

## Citing

If you use this code in your research, please use the following BibTeX entry.

`@article{CAO2022186,
title = {Unsupervised domain-share CNN for machine fault transfer diagnosis from steady speeds to time-varying speeds},
journal = {Journal of Manufacturing Systems},
volume = {62},
pages = {186-198},
year = {2022},
issn = {0278-6125},
doi = {https://doi.org/10.1016/j.jmsy.2021.11.016},
url = {https://www.sciencedirect.com/science/article/pii/S0278612521002466},
author = {Hongru Cao and Haidong Shao and Xiang Zhong and Qianwang Deng and Xingkai Yang and Jianping Xuan}
}`

