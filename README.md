# Physics-informed neural networks (Pytorch)
### Continuous time models: Burgers' equation / Allen-Cahn equation
This repository provides basic PINNs to solve continuous time models without reference data for model training. That is, this approach does not use any observation data except for initial and boundary condition data.

* This repository provides easy codes for educational purposes.
* The purpose of the repository is to set a baseline using Pytorch for our PINN project.
* Our model and the reference are not exactly the same.

[Reference] M.Raissi, P.Perdikaris, G.E.Karniadakis (2019) Physics-informed neural networks - A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics Vol 378 (2019) pages 686–707, https://doi.org/10.1016/j.jcp.2018.10.045

## 1. Execution examples
Use the **pinns.ipynb** file

### 1.1. Default setting
```python
num_t = 100 # the number of t-grids
num_x = 256 # the number of x-grids
num_epochs = 200000 # the number of epochs
num_hidden = 4 # the number of hidden layers
num_nodes = 128 # the number of nodes in each hidden layer
lr = 1e-3 # a learning rate 
eq='bg' #'bg': Burgers' equation, 'ac': Allen-Cahn equation
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
<img width="500" alt="r1" src="https://user-images.githubusercontent.com/52735725/172351462-51b0fbfb-7ed4-41e2-ae7a-efe41e0af51b.png">
<img width="500" alt="r" src="https://user-images.githubusercontent.com/52735725/172351420-65e760df-ed0d-4f4c-a934-232ff2f78bfb.png">
</p>

## 3. Reference Datasets
Source: https://github.com/maziarraissi/PINNs

