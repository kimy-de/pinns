import argparse
import torch
import torch.nn as nn
import data
import utils
import model
import numpy as np
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNs: Burgers equation and Allen-Cahn equation')
    parser.add_argument('--dt', default=2e-2, type=float, help='dt')
    parser.add_argument('--dx', default=1e-2, type=float, help='dx')
    parser.add_argument('--Nu', default=100, type=int, help='N_u')
    parser.add_argument('--pretrained', default=0, type=int, help='pretrained model')  
    parser.add_argument('--num_epochs', default=200000, type=int, help='number of epochs')  
    parser.add_argument('--num_hidden', default=9, type=int, help='number of hidden layers')   
    parser.add_argument('--num_nodes', default=20, type=int, help='number of nodes in each hidden layer')                      
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--eq', default='bg', type=str, help='bg, ac')
    args = parser.parse_args()
    print(args)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("0. Operation mode: ", device)
    
    if args.eq == 'bg':
        t_data, x_data, u_data, t_data_f, x_data_f = data.bg_generator(args.dt, args.dx, args.Nu)  
    elif args.eq == 'ac':
        t_data, x_data, u_data, t_data_f, x_data_f = data.ac_generator(args.dt, args.dx, args.Nu)
    else:
        print("There exists no the equation.")
        exit(0)
        
    variables = torch.FloatTensor(np.concatenate((t_data, x_data), 1)).to(device)
    variables_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), 1)).to(device)
    variables_f.requires_grad = True
    u_data = torch.FloatTensor(u_data).to(device)
    print("1. Generated datasets..")
    
    layer_list = [2] + args.num_hidden * [args.num_nodes] + [1]
    pinn = model.pinn(layer_list).to(device)
    if args.pretrained == 1:
        pinn.load_state_dict(torch.load('./'+args.eq+'1d.pth'))
    print("2. Completed loading a model..")
    
    print("3. Training Session")
    optimizer = torch.optim.Adam(pinn.parameters(), betas=(0.999,0.999), lr=args.lr)
    
    loss_graph = []
    ls = 1e-3
    bep = 0
    for ep in tqdm(range(args.num_epochs)):
        
        optimizer.zero_grad()
        
        # Full batch
        u_hat = pinn(variables)
        u_hat_f = pinn(variables_f)
        
        if args.eq == 'bg':
            loss_f = torch.mean(utils.burgers_equation(u_hat_f, variables_f) ** 2)
        elif args.eq == 'ac':
            loss_f = torch.mean(utils.ac_equation(u_hat_f, variables_f) ** 2)
            
        loss_u = torch.mean((u_hat - u_data) ** 2)
        loss = loss_f + loss_u
        loss.backward() 
        optimizer.step()
        
        l = loss.item()
        loss_graph.append(l)
        if l < ls:
            ls = l
            bep = ep
            torch.save(pinn.state_dict(), './'+args.eq+'1d.pth')
            
        if ep % 1000 == 0:
            print(f"Train loss: {l}") 
        
        
    print(f"[Best][Epoch: {bep}] Train loss: {ls}") 
    plt.figure(figsize=(10, 5))
    plt.plot(loss_graph)
    plt.savefig('./loss_'+args.eq+'.png')
    
    print("4. Inference Session")
    pinn.load_state_dict(torch.load('./'+args.eq+'1d.pth'))
    if args.eq == 'bg':       
        t_test, x_test = data.bg_generator(1/101, 1/256, typ='test')
        t = np.linspace(0, 1, 101).reshape(-1,1)
        x = np.linspace(-1, 1, 256).reshape(-1,1)
        T = t.shape[0]
        N = x.shape[0]
        
        test_variables = torch.FloatTensor(np.concatenate((t_test, x_test), 1)).to(device)
        with torch.no_grad():
            u_pred = pinn(test_variables)
        u_pred = u_pred.cpu().numpy().reshape(N,T)

        # reference data
        data = scipy.io.loadmat('./data/burgers_shock.mat')  
        Exact = np.real(data['usol'])  
        err = u_pred[:,:-1]-Exact
        
    elif args.eq == 'ac':
        t = np.linspace(0, 1, 201).reshape(-1,1) # T x 1
        x = np.linspace(-1, 1, 513)[:-1].reshape(-1,1) # N x 1
        T = t.shape[0]
        N = x.shape[0]
        T_star = np.tile(t, (1, N)).T  # N x T
        X_star = np.tile(x, (1, T))  # N x T
        t_test = T_star.flatten()[:, None]
        x_test = X_star.flatten()[:, None]
        
        test_variables = torch.FloatTensor(np.concatenate((t_test, x_test), 1)).to(device)
        with torch.no_grad():
            u_pred = pinn(test_variables)
        u_pred = u_pred.cpu().numpy().reshape(N,T)
        data = scipy.io.loadmat('./data/AC.mat')
        Exact = np.real(data['uu'])
        err = u_pred-Exact

    err = np.linalg.norm(err,2)/np.linalg.norm(Exact,2)   
    print(f"L2 Relative Error: {err}")

    utils.resplot(x, t, t_data, x_data, Exact, u_pred, args.eq)
    print("5. Completed")
