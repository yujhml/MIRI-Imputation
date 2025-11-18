# %% maximum mean discrepancy

import torch
import numpy as np

def mmd(X, Y, kernel='rbf', sigma=None):
    """
    Maximum Mean Discrepancy
    """
    if kernel == 'rbf':
        XX = torch.cdist(X, X)
        XY = torch.cdist(X, Y)
        YY = torch.cdist(Y, Y)
        
        if sigma is None:
            sigma = np.median(XY.detach().cpu().numpy())
        
        XX = torch.exp(-XX**2 / (2 * sigma**2)).mean()
        XY = torch.exp(-XY**2 / (2 * sigma**2)).mean()
        YY = torch.exp(-YY**2 / (2 * sigma**2)).mean()
        
        return XX + YY - 2 * XY
    else:
        raise NotImplementedError