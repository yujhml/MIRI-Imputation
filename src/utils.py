import torch
import torch.nn as nn
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

# %% utililities

def tonumpy(x):
    return x.detach().cpu().numpy()
def totorch(x):
    return torch.tensor(x, dtype = torch.float32, device=device)

def checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, filename)

def showpics3(X):
    # scatter plot
    plt.figure()
    # plot the first 16 images in X0
    for i in range(49):
        plt.subplot(7, 7, i+1)
        img = tonumpy(X[i].reshape(3, 32, 32).permute(1, 2, 0))
        plt.imshow(img.reshape(32, 32, 3))
        plt.axis('off')

def vis_score_net(model):

    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    sigma = totorch(.1) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)

    plt.subplot(1, 3, 1, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    sigma = totorch(.5) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)

    plt.subplot(1, 3, 2, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    sigma = totorch(.9) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)
    
    plt.subplot(1, 3, 3, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    display.clear_output(wait=True)
    plt.show()
    
def vis_score_net2(model, M_value):

    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    M = np.ones(100) * M_value
    XX = np.concatenate([XX, M[:, None]], 1)

    sigma = totorch(.1) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)

    plt.subplot(1, 3, 1, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    sigma = totorch(.5) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)

    plt.subplot(1, 3, 2, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    sigma = totorch(.9) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)
    
    plt.subplot(1, 3, 3, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    display.clear_output(wait=True)
    plt.show()
    
def vis_score_net3(model):

    x = np.linspace(-1, 5, 10)
    y = np.linspace(-3, 1.235, 10)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    M = np.zeros_like(XX)
    XX = np.concatenate([XX, M], 1)

    sigma = totorch(.1) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)

    plt.subplot(1, 3, 1, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    sigma = totorch(.5) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)

    plt.subplot(1, 3, 2, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    sigma = totorch(.9) 
    Z = tonumpy(model(totorch(XX), sigma))
    Z = Z.reshape(10, 10, 2)
    
    plt.subplot(1, 3, 3, aspect='equal')
    plt.quiver(X, Y, Z[:, :, 0], Z[:, :, 1])
    plt.title(f"sigma = {sigma.item():.2f}")
    
    display.clear_output(wait=True)
    plt.show()