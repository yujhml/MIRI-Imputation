
# %%
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils
from src.utils import tonumpy, totorch 
from src.mine import MINE
import numpy as np
from src.mmd import mmd
from tqdm import tqdm
from IPython import display
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from lightgbm import LGBMRegressor
from time import time


def impute_now(X0, M, Xstar, imputer_name, max_rounds = 100, clamp=False, callback = None, verbose=False, 
               batchsize = 100, maxepochs = 100, odesteps = 100):
    if imputer_name == 'miri':
        from src.cnnrgb import CNNNet as CNNNetRGB
        from src.cnn import CNNNet
        from src.imputer import rectified_impute
        from src.mlp import MLP
        print("Using miri imputer", "Tests are OK!")
        X0_miri = X0.detach().clone()
        # return rectified_impute(X0_rectified, M, Xstar, max_rounds, clamp, callback, verbose, batchsize, maxepochs, odesteps)
        # return rectified_impute_xgb(X0_rectified, M, Xstar, max_rounds, clamp, callback, verbose)
        if X0.shape[1] == 1*32*32:
            modelclass = CNNNet
            clamp = True
        elif X0.shape[1] == 3*32*32:
            modelclass = CNNNetRGB
            clamp = True
        elif X0.shape[1] == 1*64*64:
            modelclass = CNNNet
            clamp = True
        elif X0.shape[1] == 3*64*64:
            modelclass = CNNNetRGB
            clamp = True
        else:
            modelclass = MLP
            clamp = False
        X0_knn, mmd_list, mi_list, model_list = rectified_impute(X0_miri, M, Xstar, modelclass, max_rounds, clamp, callback, verbose, batchsize, maxepochs, odesteps)
        return X0_knn, mmd_list, mi_list
    
    elif imputer_name == 'knn' or imputer_name == 'iter':
        X0_knn = X0.detach().clone()
        X0_knn[M == 0] = torch.nan

        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import KNNImputer, IterativeImputer
        
        if imputer_name == 'knn':       
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = IterativeImputer(max_iter=1000, random_state=0)
        
        X0_knn = totorch(imputer.fit_transform(X0_knn.cpu().numpy())).cpu()
        
        if X0_knn.shape[0] < 15000:
            return X0_knn, torch.Tensor([[mmd(X0_knn, Xstar)]]), torch.Tensor([MINE(X0_knn, M)[0]])
        else:
            return X0_knn, torch.Tensor([[mmd(X0_knn[:15000, :], Xstar[:15000, :])]]), torch.Tensor([MINE(X0_knn, M)[0]])
    
    elif imputer_name == 'knewimp':
        from src.competitors.wgf_imp import NeuralGradFlowImputer
        X0 = X0.detach().clone()
        # X0X0 = torch.cdist(X0, X0)
        # sigma = np.median(X0X0.detach().cpu().numpy())
        # print("bandwidth chosen as median: ", sigma)
        # sigma = 1.0
        sigma = 0.1
        X0[M==0] = torch.nan
        imputer = NeuralGradFlowImputer(entropy_reg=10.0, bandwidth=sigma, device = device,
                                      score_net_epoch=100, niter=2,
                                      initializer=None, mlp_hidden=[256, 256], lr=.1,
                                      score_net_lr=1.0e-3)
        X0_knn = totorch(imputer.fit_transform(X0, X_true = Xstar, clamp=clamp)[0]).cpu()
        if X0_knn.shape[0] < 15000:
            return X0_knn, torch.Tensor([[mmd(X0_knn, Xstar)]]), torch.Tensor([MINE(X0_knn, M)[0]])
        else:
            return X0_knn, torch.Tensor([[mmd(X0_knn[:15000, :], Xstar[:15000, :])]]), torch.Tensor([MINE(X0_knn, M)[0]])
    
    elif imputer_name == 'gain_image':
        from src.competitors.gain_imp import gain_impute
        X0 = X0.detach().clone()
        X0[M == 0] = torch.nan
        # reshape X0 to (N, C, H, W)
        if X0.shape[1] == 3*32*32:
            X0 = X0.reshape(X0.shape[0], 3, 32, 32)
        elif X0.shape[1] == 3*64*64:
            X0 = X0.reshape(X0.shape[0], 3, 64, 64)
        elif X0.shape[1] == 1*32*32:
            X0 = X0.reshape(X0.shape[0], 1, 32, 32)
        elif X0.shape[1] == 1*64*64:
            X0 = X0.reshape(X0.shape[0], 1, 64, 64)
        
        X0_knn = gain_impute(X0)
        X0_knn = X0_knn.reshape(X0_knn.shape[0], -1).cpu()
        
        if X0_knn.shape[0] < 15000:
            return X0_knn, torch.Tensor([[mmd(X0_knn, Xstar)]]), torch.Tensor([MINE(X0_knn, M)[0]])
        else:
            return X0_knn, torch.Tensor([[mmd(X0_knn[:15000, :], Xstar[:15000, :])]]), torch.Tensor([MINE(X0_knn, M)[0]])
        
    elif imputer_name == 'missdiff':
        from src.competitors.recdiff_imp_tabular import missdiff_impute
        X0 = X0.detach().clone()
        X0[M == 0] = torch.nan
        
        X0_knn = missdiff_impute(X0)
        X0_knn = X0_knn.reshape(X0_knn.shape[0], -1).cpu()
        
        if X0_knn.shape[0] < 15000:
            return X0_knn, torch.Tensor([[mmd(X0_knn, Xstar)]]), torch.Tensor([MINE(X0_knn, M)[0]])
        else:
            return X0_knn, torch.Tensor([[mmd(X0_knn[:15000, :], Xstar[:15000, :])]]), torch.Tensor([MINE(X0_knn, M)[0]])
    else: 
        X0_knn = X0.detach().clone()
        X0_knn[M == 0] = torch.nan
        
        from hyperimpute.plugins.imputers import Imputers
        imputer = Imputers().get(imputer_name)

        df = imputer.fit_transform(tonumpy(X0_knn).astype(np.float64))
        
        X0_knn = totorch(df.to_numpy()).cpu()
        
        if X0_knn.shape[0] < 15000:
            return X0_knn, torch.Tensor([[mmd(X0_knn, Xstar)]]), torch.Tensor([MINE(X0_knn, M)[0]])
        else:
            return X0_knn, torch.Tensor([[mmd(X0_knn[:15000, :], Xstar[:15000, :])]]), torch.Tensor([MINE(X0_knn, M)[0]])

