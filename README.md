# Physics-informed neural networks (Pytorch)
### Continuous time models: Burgers' equation / Allen-Cahn equation
This repository provides basic PINNs to solve continuous time models without training data. That is, this approach does not use any observation data except for initial and boundary condition data. It is challenging to optimize PINNs for complicated PDEs. Using the reference model, good accuracy is achieved for a Burgers' equation, but the inference for an Allen-Cahn equation causes high relative errors compared to that of the Burgers' equation. Thus, diverse optimization approaches and different network architectures have been developed to improve the reference approach.

* The purpose of the repository is to set a baseline using Pytorch for our PINN project.
* Our model and the reference are not exactly the same. (improved)

[Reference] M.Raissi, P.Perdikaris, G.E.Karniadakis (2019) Physics-informed neural networks - A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics Vol 378 (2019) pages 686–707, https://doi.org/10.1016/j.jcp.2018.10.045

## 1. Execution examples
```bash
$ python main.py --eq 'bg' --num_hidden 9 --num_nodes 20 --dx 1/256 --dt 0.002                                 
```
```bash
$ python main.py --eq 'ac' --num_hidden 4 --num_nodes 200 --dx 1/256 --dt 0.002                                    
```
### 1.1. Default setting
```
Nu=100, dt=0.02, dx=0.01, lr=0.001, num_epochs=200000, num_hidden=9, num_nodes=20 pretrained=0 eq='bg'
```
### 1.2. Outputs
L2 relative error, a training loss graph, and two result figures

## 2. Results
### 2.1. Burgers' equation
<p align="center">
<img width="500" alt="r1" src="https://user-images.githubusercontent.com/52735725/172022015-09c094e7-ae69-44fd-9319-b5039e436f15.png">
<img width="500" alt="r" src="https://user-images.githubusercontent.com/52735725/164943040-a356729e-795e-42ed-b37a-9abf6fa8bb46.png">
</p>

### 2.2. Allen-Cahn equation
<p align="center">
<img width="500" alt="r1" src="https://user-images.githubusercontent.com/52735725/172065924-8d884678-ced7-4b35-9787-3f934b27870f.png">
<img width="500" alt="r" src="https://user-images.githubusercontent.com/52735725/172065910-e21e943a-1200-4871-ad3d-a78b9ce49dd1.png">
</p>

## 3. Datasets
Source: https://github.com/maziarraissi/PINNs

