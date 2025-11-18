import torch
import torch.nn as nn
import torch.optim as optim
from time import time
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from IPython.display import clear_output
import matplotlib.pyplot as plt
from src.mmd import mmd
from src.mine import MINE
import copy

@torch.no_grad()
def sample_ode(model, z0=None, N=None, clamp=False):
    d = z0.shape[1] // 3
    
    dt = 1./N
    
    for i in range(N):
      t = torch.tensor(i / N, device=device)
      t = t.expand(z0.shape[0], 1)

      pred = model(z0, t)
      z0[:, :d] = z0[:, :d] + pred * dt
      z0[:, d:2*d] = z0[:, d:2*d] + pred * dt
      
      if clamp:
          z0[:, :d] = torch.clamp(z0[:, :d], 0, 1)
      
    return z0[:, :d]

def rectified_impute(X0, M, Xstar, modelclass, max_rounds = 100, clamp = False, callback = None, verbose=False, 
                    batchsize = 100, maxepochs = 100, odesteps = 100, dsetname = "NULL"):
    d = X0.shape[1] 
    n = X0.shape[0]
    criterion = nn.MSELoss()
    print(f'n: {n}, d: {d}, batchsize: {batchsize}')
    
    X1 = X0[torch.randperm(X0.shape[0]), :].clone()
    
    mi_list = []
    mmd_list = []
    model_list = []
    try:
        for round in range(max_rounds):
            model = modelclass(d).to(device)
            
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            if n > 10000: 
                mmd_round = mmd(X0[:10000, :], Xstar[:10000, :])
            else: 
                mmd_round = mmd(X0, Xstar)
            print(f'round: {round+1}, mmd: {mmd_round:.5f}')
            mmd_list.append(mmd_round)

            # estimate Mutual information between X0 and M
            print("Estimating mutual information ...")
            mi = MINE(X0, M)[0]
            # mi = 0
            mi_list.append(mi)
            print(f'mi: {mi}')
            
            print("Training vector field ...")
            
            tensorset = torch.utils.data.TensorDataset(torch.cat([X0, M], dim=1), X1)
            trainloader = torch.utils.data.DataLoader(tensorset, batch_size=batchsize, shuffle=True)

            starttime = time()
            for epoch in tqdm(range(maxepochs)):
                for iter, (Z0i, X1i) in enumerate(trainloader):
                    X1i = X1i.to(device)
                    X0i = Z0i[:, :d].to(device)
                    Mi = Z0i[:, d:].to(device)
                    
                    t = torch.rand(X1i.shape[0], 1, device=device) # sample t from [0, 1]
                    Xti = t * X1i + (1-t) * X0i
                    Xti1 = Xti.clone()
                    Xti1[Mi == 1] = X0i[Mi == 1]
                    Xti2 = Xti.clone()
                    Xti2[Mi == 1] = X1i[Mi == 1]

                    Zti = torch.cat([Xti1, Xti2, Mi], dim=1)
                    pred = model(Zti, t)
                    y = X1i - X0i
                    loss = criterion(pred[Mi==0], y[Mi==0]) 
                    
                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            if verbose:
                # print(f"epoch: {epoch}, loss: {loss.item():.5f}")
                pass

                        
            model.eval()
            Z0 = torch.cat([X0, X0, M], dim=1)
            
            if verbose:
                print("sampling from the model (solving ODE) ...")
            
            # impute in batches
            imputebatch = 500
            for i in tqdm(range(0, n, imputebatch)):
                Z0i = Z0[i:i+imputebatch, :].clone().to(device)
                
                # sample from the model
                Z1i = sample_ode(model, z0=Z0i, N=odesteps, clamp=clamp).to('cpu')
                
                X0[i:i+imputebatch][M[i:i+imputebatch, :] == 0] = Z1i[M[i:i+imputebatch, :] == 0]
            
            X1 = X0[torch.randperm(X0.shape[0]), :].clone()
                        
            print(f'Finished round: {round+1} / {max_rounds}, time: {time() - starttime}')
            print("")

            model_list.append(model)
            # save checkpoint
            torch.save([X0, mmd_list, mi_list, model_list], "./%s_rec.pt" % dsetname)
        
    except KeyboardInterrupt:
        print("Interrupted, returning the current imputed data")
        pass
    
    return X0, mmd_list, mi_list, model_list