# Physics-informed neural networks (Pytorch)
### Continuous time models: Burgers' equation / Allen-Cahn equation
This repository provides basic PINNs to solve continuous time models without reference data for model training. That is, this approach does not use any observation data except for initial and boundary condition data. It is challenging to optimize PINNs for complicated PDEs. Using the baseline model, good accuracy is achieved for a Burgers' equation (relative error: 9.98e-04), but the inference for an Allen-Cahn equation (relative error: 1.55e-2) causes high relative errors compared to that of the Burgers' equation.

* The purpose of the repository is to set a baseline using Pytorch for our PINN project.
* Our model and the reference are not exactly the same. (improved)

[Reference] M.Raissi, P.Perdikaris, G.E.Karniadakis (2019) Physics-informed neural networks - A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics Vol 378 (2019) pages 686–707, https://doi.org/10.1016/j.jcp.2018.10.045

## 1. Execution examples
```bash
$ python main.py --eq 'bg' --num_hidden 9 --num_nodes 20 --dx 0.0039 --dt 0.01                                 
```
```bash
$ python main.py --eq 'ac' --num_hidden 4 --num_nodes 200 --dx 0.0039 --dt 0.01                                    
```
### 1.1. Default setting
```
Nu: the number of initial condition data
dt, dx: mesh sizes
num_epochs, lr: the number of epochs, a learning rate 
num_hidden, num_nodes: number of hidden layers, number of nodes in each hidden layer 
pretrained: 0: an initialized model, 1: a pretrained model
eq: 'bg': Burgers' equation, 'ac': Allen-Cahn equation
```
### 1.2. Outputs
L2 relative error, a training loss graph, and two result figures

## 2. Results
### 2.1. Burgers' equation
<p align="center">
<img width="500" alt="r1" src="https://user-images.githubusercontent.com/52735725/172101128-a5aedd20-a871-42ef-b748-9cae98fc1ab0.png">
<img width="500" alt="r" src="https://user-images.githubusercontent.com/52735725/164943040-a356729e-795e-42ed-b37a-9abf6fa8bb46.png">
</p>

### 2.2. Allen-Cahn equation
<p align="center">
<img width="500" alt="r1" src="https://user-images.githubusercontent.com/52735725/172295341-3a57246b-75d5-49aa-a553-70aea9be8df3.png">
<img width="500" alt="r" src="https://user-images.githubusercontent.com/52735725/172295382-e49def99-b90a-45ba-9af8-dfb93d7fbe99.png">
</p>

## 3. Reference Datasets
Source: https://github.com/maziarraissi/PINNs




