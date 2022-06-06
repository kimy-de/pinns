import torch
import numpy as np
import matplotlib.pyplot as plt

def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0]
    return derivative

def burgers_equation(u, tx):
    u_tx = fwd_gradients(u, tx)
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    u_xx = fwd_gradients(u_x, tx)[:, 1:2]
    e = u_t + u*u_x - (0.01/np.pi)*u_xx
    return e

def ac_equation(u, tx):
    u_tx = fwd_gradients(u, tx)
    u_t = u_tx[:, 0:1]
    u_x = u_tx[:, 1:2]
    u_xx = fwd_gradients(u_x, tx)[:, 1:2]
    e = u_t -0.0001*u_xx + 5*u**3 - 5*u
    return e 

def resplot(x, t, t_data, x_data, Exact, u_pred, eq):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x, Exact[:,0],'-')
    plt.plot(x, u_pred[:,0],'--')
    plt.legend(['Actual', 'Prediction'])
    plt.title("Initial condition ($t=0$)")
    
    plt.subplot(2, 2, 2)
    t_step = int(0.25*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    plt.legend(['Actual', 'Prediction'])
    plt.title("$t=0.25$")
    
    plt.subplot(2, 2, 3)
    t_step = int(0.5*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    plt.legend(['Actual', 'Prediction'])
    plt.title("$t=0.5$")
    
    plt.subplot(2, 2, 4)
    t_step = int(0.99*len(t))
    plt.plot(x, Exact[:,t_step],'-')
    plt.plot(x, u_pred[:,t_step],'--')
    plt.legend(['Actual', 'Prediction'])
    plt.title("$t=0.99$")
    plt.savefig('./inference_'+eq+'.png')
    
    plt.figure(figsize=(10, 5))
    plt.imshow(u_pred, interpolation='nearest', cmap='jet',
               extent=[t.min(), t.max(), x.min(), x.max()],
               origin='lower', aspect='auto')
    plt.clim(-1, 1)
    plt.ylim(-1,1)
    plt.xlim(0,1)
    plt.scatter(t_data, x_data)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('u(t,x)')
    plt.savefig('./evolution_'+eq+'.png')
