# Physics-informed neural networks (Pytorch)
### Data-driven solution of Burgers' equation - continuous time models

[Reference] M.Raissi, P.Perdikaris, G.E.Karniadakis (2019) Physics-informed neural networks - A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics Vol 378 (2019) pages 686–707, https://doi.org/10.1016/j.jcp.2018.10.045

## Execution
```bash
$ python main.py                                   
```
### Default setting
```
Nu=100, dt=0.02, dx=0.01, lr=0.001, num_epochs=0, num_hidden=9, num_nodes=20
```
## Results
<p align="center">
<img width="500" alt="r1" src="https://user-images.githubusercontent.com/52735725/164942859-f83123bb-668e-44be-b2fc-514a430be416.png">
<img width="800" alt="r" src="https://user-images.githubusercontent.com/52735725/164943040-a356729e-795e-42ed-b37a-9abf6fa8bb46.png">
</p>

## To do
- [ ] Add a numerical method
- [ ] Add other PDEs
