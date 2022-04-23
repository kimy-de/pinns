import argparse
import torch
import torch.nn as nn
import data
import utils
import model
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN: Burgers equation')
    parser.add_argument('--dt', default=2e-2, type=float, help='dt')
    parser.add_argument('--dx', default=1e-2, type=float, help='dx')
    parser.add_argument('--Nu', default=100, type=int, help='N_u')
    parser.add_argument('--num_epochs', default=20000, type=int, help='number of epochs')  
    parser.add_argument('--num_hidden', default=9, type=int, help='number of hidden layers')   
    parser.add_argument('--num_nodes', default=20, type=int, help='number of nodes in each hidden layer')                      
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()
    print(args)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("0. Operation mode: ", device)
    
    t_data, x_data, u_data, t_data_f, x_data_f = data.bg_generator(args.dt, args.dx, args.Nu)  
    variables = torch.FloatTensor(np.concatenate((t_data, x_data), 1)).to(device)
    variables_f = torch.FloatTensor(np.concatenate((t_data_f, x_data_f), 1)).to(device)
    variables_f.requires_grad = True
    u_data = torch.FloatTensor(u_data).to(device)
    print("1. Generated datasets..")
    
    layer_list = [2] + args.num_hidden * [args.num_nodes] + [1]
    pinn = model.pinn(layer_list).to(device)
    print("2. Completed loading a model..")
    
    print("3. Training Session")
    optimizer = torch.optim.Adam(pinn.parameters(), lr=args.lr)
    
    loss_graph = []
    ls = 1
    bep = 0
    for ep in tqdm(range(args.num_epochs)):
        
        optimizer.zero_grad()
        
        # Full batch
        u_hat = pinn(variables)
        u_hat_f = pinn(variables_f)
        
        loss_f = torch.mean(utils.burgers_equation(u_hat_f, variables_f) ** 2)
        loss_u = torch.mean((u_hat - u_data) ** 2)
        loss = loss_f + loss_u
        loss.backward() 
        optimizer.step()
        
        l = loss.item()
        loss_graph.append(l)
        if l < ls:
            ls = l
            bep = ep
            torch.save(pinn.state_dict(), './bg1d.pth')
            
        if ep % 1000 == 0:
            print(f"Train loss: {l}") 
        
        
    print(f"[Best][Epoch: {bep}] Train loss: {ls}") 
    plt.figure(figsize=(10, 5))
    plt.plot(loss_graph)
    plt.savefig('./loss_result.png')
    
    print("4. Inference Session")
    pinn.load_state_dict(torch.load('./bg1d.pth'))
    t_test, x_test = data.bg_generator(1e-3, 1e-2, typ='test')
    t = np.linspace(0, 1, 1000).reshape(-1,1)
    x = np.linspace(-1, 1, 100).reshape(-1,1)
    T = t.shape[0]
    N = x.shape[0]
    
    test_variables = torch.FloatTensor(np.concatenate((t_test, x_test), 1)).to(device)
    with torch.no_grad():
        u_pred = pinn(test_variables)
    
    u_pred = u_pred.cpu().numpy().reshape(N,T)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x, u_pred[:,0],'.-')
    plt.title("Initial condition ($t=0$)")
    
    plt.subplot(2, 2, 2)
    t_step = int(0.3*len(t))
    plt.plot(x, u_pred[:,t_step],'.-')
    plt.title("Initial condition ($t=%.3f$)" %(1e-3*t_step))
    
    plt.subplot(2, 2, 3)
    t_step = int(0.5*len(t))
    plt.plot(x, u_pred[:,t_step],'.-')
    plt.title("Initial condition ($t=%.3f$)" %(1e-3*t_step))
    
    plt.subplot(2, 2, 4)
    t_step = int(0.75*len(t))
    plt.plot(x, u_pred[:,t_step],'.-')
    plt.title("Initial condition ($t=%.3f$)" %(1e-3*t_step))
    plt.savefig('./inference.png')
    
    plt.figure(figsize=(10, 5))
    plt.imshow(u_pred, interpolation='nearest', cmap='jet',
               extent=[t.min(), t.max(), x.min(), x.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.scatter(t_data, x_data)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('u(t,x)')
    plt.savefig('./inference_result.png')
    print("5. Completed")
        