from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union

# Define DEVICE consistently like in GAIN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MissDiffImage:
    """MissDiff model for image data with missing values."""

    def __init__(
        self,
        inputdim=1,
        lr = 1e-4,
        num_diffusion_steps=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device

        # load the network
        if inputdim == 32 * 32 * 3 or inputdim == 3 * 64 * 64:
            from src.cnnrgb import CNNNet2_T
            self.network = CNNNet2_T(inputdim).to(device)
        else:
            from src.mlp import MLP2
            self.network = MLP2(inputdim).to(device)

        # the optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    # t X_1 + (1-t) X_0
    def forward_diffusion(self, x_1, t):
        if x_1.shape[1] == 3 * 64 * 64 or x_1.shape[1] == 32 * 32 * 3:
            # generate mixture of X_0 and noise
            noise = torch.rand_like(x_1)
        else:
            noise = torch.randn_like(x_1)
        # reshape t to match the batch size
        t = t.view(-1, 1)
        x_t = t * x_1 + (1-t) * noise
        return x_t, noise

    def fit(self, X, M, n_epochs=100, batch_size=64, verbose=1):
        X1 = X.detach().clone(); 
        
        dataset = TensorDataset(X1, M) 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.MSELoss()

        # Training loop
        if verbose > 0:
            print(f"Training MissDiff model for {n_epochs} epochs...")

        for epoch in tqdm(range(n_epochs)):
            epoch_losses = []

            for x1_batch, mask_batch in dataloader: # Get both data and mask
                # Move data to device, one batch at a time
                x1_batch = x1_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                
                self.optimizer.zero_grad()
                batch_size = x1_batch.shape[0]  # Fix: Get the actual batch size (first dimension only)

                # Sample random timesteps
                t = torch.rand(batch_size, device=self.device) 

                # Forward process X_t = t * X_1 + (1-t) * X_0
                x_t, noise = self.forward_diffusion(x1_batch, t)
                noise_pred = self.network(x_t, t.unsqueeze(1))  
                noise_pred = noise_pred * mask_batch  
                output = (x1_batch - noise) * mask_batch

                # MSE criterion 
                loss = criterion(noise_pred, output)  

                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            if verbose > 0 and (epoch + 1) % (max(n_epochs // 10, 1)) == 0:
                tqdm.write(f"Epoch {epoch+1}/{n_epochs}, Loss: {np.mean(epoch_losses):.6f}")

        return self

    @torch.no_grad()
    def sample(self, X0, mask, num_samples=1, clamp=False):
        
        X0 = X0.detach().clone().to(self.device)
        mask = mask.detach().clone().to(self.device)
        self.network.eval()
        
        imp_list = []
        for i in tqdm(range(num_samples)):
            if X0.shape[1] == 3 * 64 * 64 or X0.shape[1] == 32 * 32 * 3:
                X0[mask == 0] = torch.rand_like(X0)[mask == 0]  # Set missing values to random values
            else:
                X0[mask == 0] = torch.randn_like(X0)[mask == 0]
        
            x_t = torch.randn_like(X0)
            for t in range(self.num_diffusion_steps):
                t = t / self.num_diffusion_steps * torch.ones(x_t.shape[0], device=self.device)  # Create a tensor of the same size as x_t
                
                # Sample from the model
                xt_obs = self.forward_diffusion(X0, t)[0]  # Get x0_t from forward diffusion
                
                # part of the vector is specified using the forward process
                x_t = x_t * (1-mask) + xt_obs * mask 
                pred = self.network(x_t, t.unsqueeze(1)) 
                
                # run one step of rectified flow ODE
                x_t = x_t + 1/self.num_diffusion_steps * pred
                if clamp:
                    x_t = torch.clamp(x_t, 0, 1)
                
            x_imp = x_t * (1-mask) + xt_obs * mask    
            imp_list.append(x_imp)
        
        x_ave = torch.zeros_like(x_imp)
        for i in range(len(imp_list)):
            x_ave += imp_list[i]
        
        x_ave /= len(imp_list)
        
        print("amount of updates:" , torch.mean((x_ave * (1-mask) - X0 * (1-mask)) ** 2).item())
        return x_ave


def missdiff_impute(
    X: torch.Tensor,
    n_epochs: int = 100,
    n_iter: int = 100,
    n_samples: int = 5,
    batch_size: int = 500,
    inference_batch_size: int = 500,  
    learning_rate: float = 1e-2,
    verbose: int = 1
):
    """
    Impute missing values in images using MissDiff.
    Args:
        X0: Input tensor with missing values (NaN) -
        n_epochs: Number of training epochs
        n_iter: Number of diffusion steps
        n_samples: Number of samples to average for imputation
        batch_size: Batch size for training
        inference_batch_size: Maximum batch size for inference to control memory usage
        learning_rate: Learning rate

    Returns:
        Imputed data tensor
    """
    
    # Get missing mask (1 for observed, 0 for missing)
    M = (~torch.isnan(X)).float()
    # set missign to random values. randn for tabular, rand for images
    X[torch.isnan(X)] = torch.zeros_like(X)[torch.isnan(X)]

    # Initialize and train model
    if verbose > 0:
        print(f"Training MissDiff model with {n_epochs} epochs and {n_iter} diffusion steps...")

    model = MissDiffImage(
        inputdim=X.shape[1],
        lr = learning_rate,
        num_diffusion_steps=n_iter,
        device=DEVICE
    )

    model.fit(
        X, M, # Pass both X0 and M to fit
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # Generate samples
    if verbose > 0:
        print(f"Generating {n_samples} samples for imputation...")

    # Clean up memory before sampling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # do imputation 500 samples a time 
    X_imputed = torch.zeros_like(X)
    for i in range(0, X.shape[0], inference_batch_size):
        # Get the current batch
        batch_X0 = X[i:i + inference_batch_size]
        batch_M = M[i:i + inference_batch_size]

        clamp = False
        # Sample from the model
        if X.shape[1] == 3 * 64 * 64 or X.shape[1] == 32 * 32 * 3:
            clamp = True
        X_sampled = model.sample(batch_X0, batch_M, num_samples=n_samples, clamp=clamp)

        # Average the samples
        X_imputed[i:i + inference_batch_size] = X_sampled

    # Clean up memory after sampling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X_imputed