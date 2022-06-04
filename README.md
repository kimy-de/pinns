# Physics-informed neural networks (Pytorch)
### Data-driven solution of Burgers' equation - continuous time models
[Reference] M.Raissi, P.Perdikaris, G.E.Karniadakis (2019) Physics-informed neural networks - A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics Vol 378 (2019) pages 686–707, https://doi.org/10.1016/j.jcp.2018.10.045

## 1. Execution
```bash
$ python main.py                                   
```
### 1.1. Default setting
```
Nu=100, dt=0.02, dx=0.01, lr=0.001, num_epochs=0, num_hidden=9, num_nodes=20
```
### 1.2. Outputs
L2 relative error, loss_result.png, inference.png, and inference_result.png

## 2. Results
<p align="center">
<img width="500" alt="r1" src="https://user-images.githubusercontent.com/52735725/172021813-93ba83ad-4443-4bb2-94b1-4c2584bb6490.png">
<img width="700" alt="r" src="https://user-images.githubusercontent.com/52735725/164943040-a356729e-795e-42ed-b37a-9abf6fa8bb46.png">
</p>

## To do
- [x] Add a reference dataset [source: https://github.com/maziarraissi/PINNs]
- [ ] Add other PDEs
