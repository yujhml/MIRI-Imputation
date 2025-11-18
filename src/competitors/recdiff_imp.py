import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union

# Import utility functions
# from utils.image_utils import detect_image_shape
from src.competitors.nets import CNN_T

# Define DEVICE consistently like in GAIN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MissDiffImage:
    """MissDiff model for image data with missing values."""

    def __init__(
        self,
        img_channels=1,
        img_size=32,
        hidden_dims=[64],
        num_diffusion_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Network architecture parameters
        activation: str = 'relu',
        dropout_rate: float = 0.0
    ):
        self.img_channels = img_channels
        self.img_size = img_size
        self.img_flat_size = img_channels * img_size * img_size
        self.hidden_dims = hidden_dims
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device

        # Set up noise schedule.
        self.beta = torch.linspace(beta_start, beta_end, num_diffusion_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        print(hidden_dims)
        # self.network = CNN_T(
        #     # MODIFICATION: Input dimension is now just img_channels, not img_channels * 2
        #     input_dim=img_channels,
        #     hidden_dims=hidden_dims,
        #     output_dim=img_channels,
        #     spatial_dim=img_size,
        #     activation=activation,
        #     dropout_rate=dropout_rate
        # ).to(device)
        from src.cnnrgb import CNNNet3
        print(img_size)
        self.network = CNNNet3(img_size*img_size*3).to(device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)

    def forward_diffusion(self, x_0, mask, t):
    
        # generate mixture of X_0 and noise
        noise = torch.randn_like(x_0)
        # reshape t to match the batch size
        t = t.view(-1, 1, 1, 1)
        x_t = t * x_0 + (1-t) * noise
        return x_t, noise

    def fit(self, X, M, lr=1e-4, n_epochs=100, batch_size=64, verbose=1, save_dir=None):
        # Make sure inputs are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float()
        if not isinstance(M, torch.Tensor):
            M = torch.tensor(M).float()

        X = X.to(self.device)
        M = M.to(self.device) # Mask M is still needed for the loss calculation

        # Replace NaNs with zeros for training (standard preprocessing) [1]
        X_input = X.clone()

        # Set optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Create dataset and dataloader
        dataset = TensorDataset(X_input, M) # Pass mask M along with data X
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        if verbose > 0:
            print(f"Training MissDiff model for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            epoch_losses = []

            for x_batch, mask_batch in dataloader: # Get both data and mask
                self.optimizer.zero_grad()
                batch_size = x_batch.shape[0]  # Fix: Get the actual batch size (first dimension only)

                # Sample random timesteps
                t = torch.rand(batch_size, device=self.device) 

                # Forward diffusion 
                x_t, noise = self.forward_diffusion(x_batch, mask_batch, t)
                noise_pred = self.network(x_t, t)

                # Compute loss only on observed values using the mask [1]
                loss = F.mse_loss(noise_pred * mask_batch, (x_batch - noise) * mask_batch, reduction='mean')

                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            if verbose > 0 and (epoch + 1) % (max(n_epochs // 10, 1)) == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {np.mean(epoch_losses):.6f}")

        return self

    @torch.no_grad()
    def sample(self, X0, mask, num_samples=1):
        print(".")
        # fitted score model to do impainting using X0 and the mask M, 1 is observed, 0 is missing
        
        # X0: (N, C, H, W), mask: (N, C, H, W)
        self.network.eval()
        
        x_t = torch.randn_like(X0)
        for t in range(self.num_diffusion_steps):
            t = t * torch.ones(x_t.shape[0], device=self.device) / self.num_diffusion_steps  # Create a tensor of the same size as x_t
            # print(t[0].item(), 1/self.num_diffusion_steps)
            # Sample from the model
            x0_t = self.forward_diffusion(X0, mask, t)[0]  # Get x0_t from forward diffusion
            
            x_t = x_t * (1-mask) + x0_t * mask 
            pred = self.network(x_t, t)

            # run one step of rectified flow ODE
            x_t = x_t + 1/self.num_diffusion_steps * pred
            x_t = torch.clamp(x_t, 0, 1)  # Ensure pixel values are in [0, 1]
            
        x_imp = x_t * (1-mask) + x0_t * mask    
        print("amount of updates:" , torch.sum((x_t - X0) ** 2).item())
        
        return x_imp


def missdiff_impute(
    X0: torch.Tensor,
    n_epochs: int = 10,
    n_iter: int = 100,
    n_samples: int = 5,
    batch_size: int = 100,
    inference_batch_size: int = 500,  # Add parameter for inference batch size
    learning_rate: float = 1e-2,
    hidden_dims: List[int] = [64],
    activation: str = 'relu',
    dropout_rate: float = 0.0,
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
        hidden_dims: CNN hidden dimensions as a list
        activation: Activation function
        dropout_rate: Dropout rate
        verbose: Verbosity level

    Returns:
        Imputed data tensor
    """
    # Check if input is a 4D tensor (N, C, H, W)
    if len(X0.shape)!= 4:
        raise ValueError("Input tensor must be 4D (N, C, H, W).")

    # Move tensor to device
    X0 = X0.to(DEVICE)

    # Get missing mask (1 for observed, 0 for missing)
    M = (~torch.isnan(X0)).float().to(DEVICE)
    
    X0[torch.isnan(X0)] = torch.rand_like(X0)[torch.isnan(X0)]

    # Initialize and train model
    if verbose > 0:
        print(f"Training MissDiff model with {n_epochs} epochs and {n_iter} diffusion steps...")

    model = MissDiffImage(
        img_channels=X0.shape[1],
        img_size=X0.shape[2],
        hidden_dims=hidden_dims,
        num_diffusion_steps=n_iter,
        device=DEVICE,
        activation=activation,
        dropout_rate=dropout_rate
    )

    model.fit(
        X0, M, # Pass both X0 and M to fit
        lr=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=verbose,
        save_dir=None
    )

    # Generate samples
    if verbose > 0:
        print(f"Generating {n_samples} samples for imputation...")

    # Clean up memory before sampling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pass X0 and M to sample for potential final conditioning
    X_input = X0.clone().to(DEVICE)
    # do imputation 500 samples a time 
    X_imputed = torch.zeros_like(X_input).to(DEVICE)
    for i in range(0, X_input.shape[0], inference_batch_size):
        # Get the current batch
        batch_X0 = X_input[i:i + inference_batch_size]
        batch_M = M[i:i + inference_batch_size]

        # Sample from the model
        X_sampled = model.sample(batch_X0, batch_M, num_samples=n_samples)

        # Average the samples
        X_imputed[i:i + inference_batch_size] = X_sampled

    # Clean up memory after sampling
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Replace any remaining NaNs with zeros (optional safeguard)
    if torch.isnan(X_imputed).any():
        X_imputed = torch.nan_to_num(X_imputed, nan=0.0)

    # Update missing entries only using the final imputed result
    # Note: The sample method already performs final conditioning if kept
    X0_impute = X0.clone()
    X0_impute[M == 0] = X_imputed[M == 0] # Ensure only missing parts are updated

    return X0_impute