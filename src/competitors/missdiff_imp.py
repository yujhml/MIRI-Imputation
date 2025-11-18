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

        # print(hidden_dims)
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
        # Extract alpha values for the given timestep - reshape for broadcasting with 4D tensors
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        # Sample noise (same shape as input)
        noise = torch.randn_like(x_0)

        # Apply standard forward diffusion to the whole image x_0
        # The mask 'm' is NOT used here as per MissDiff paper [1]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        # MODIFICATION REMOVED: The following lines modified x_t based on the mask,
        # which deviates from MissDiff's forward process.
        # # Replace missing values with pure noise
        # missing_mask = 1 - mask
        # x_t = x_t * mask + noise * missing_mask

        # Return the standard noisy sample and the noise used
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
        X_input[torch.isnan(X_input)] = 0

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
                t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)

                # Forward diffusion (standard process, mask not used inside)
                x_t, noise = self.forward_diffusion(x_batch, mask_batch, t)

                # Predict noise using the network
                t_scaled = t.float() / self.num_diffusion_steps
                noise_pred = self.network(x_t, t_scaled.unsqueeze(1))

                # Compute loss only on observed values using the mask [1]
                # This implementation correctly applies the mask to the error before squaring
                obs_mask_sum = mask_batch.sum() + 1e-6
                loss = F.mse_loss(noise_pred * mask_batch, noise * mask_batch, reduction='sum') / obs_mask_sum
                # Original calculation was slightly different but achieved the same goal:
                # loss = F.mse_loss(noise_pred * mask_batch, noise * mask_batch) * mask_batch.numel() / obs_mask_sum

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
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
        
        sample_list = []
        for i in range(num_samples):
            x_t = torch.randn_like(X0) # Sample from standard normal distribution
            for t in reversed(range(self.num_diffusion_steps)):
                # sample from the reverse process 
                t_tensor = torch.full((X0.shape[0],), t, device=self.device, dtype=torch.long)
                t_scaled = t_tensor.float() / self.num_diffusion_steps
                
                # Apply forward diffusion to get x0_t
                x0_t, noise = self.forward_diffusion(X0, mask, t_tensor)
                
                # Predict noise using the network
                x_t = x_t * (1-mask) + x0_t * mask # Use the mask to combine
                noise_pred = self.network(x_t, t_scaled.unsqueeze(1))
                
                # run one step of reverse diffusion
                alpha_t = self.alpha[t]
                alpha_bar_t = self.alpha_bar[t]
                beta_t = self.beta[t]
                alpha_bar_prev = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=self.device)
                
                noise = torch.randn_like(x_t) # Sample new noise
                x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_prev)) * noise_pred) + \
                    torch.sqrt((1 - alpha_bar_t) / (1 - alpha_bar_prev)) * noise

            x_t = x_t * (1-mask) + X0 * mask # Combine with original data using the mask
            sample_list.append(x_t)
        
        # average over sample list 
        ave = torch.zeros_like(sample_list[0])
        for i in range(num_samples):
            ave += sample_list[i]
        ave /= num_samples
        
        return ave   

    def save(self, path):
        """Save model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get activation and dropout from the actual network instance
        activation_used = 'relu' # Default, attempt to get actual value
        dropout_rate_used = 0.0 # Default, attempt to get actual value
        if hasattr(self.network, 'cnn') and hasattr(self.network.cnn, 'activation'):
             activation_used = self.network.cnn.activation
        if hasattr(self.network, 'cnn') and hasattr(self.network.cnn, 'dropout_rate'):
             dropout_rate_used = self.network.cnn.dropout_rate

        torch.save({
            'network': self.network.state_dict(),
            'img_channels': self.img_channels,
            'img_size': self.img_size,
            'hidden_dims': self.hidden_dims,
            'num_diffusion_steps': self.num_diffusion_steps,
            # Save the actual activation/dropout used by the network instance
            'activation': activation_used,
            'dropout_rate': dropout_rate_used,
        }, path)

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device) # Ensure loading to correct device
        self.img_channels = checkpoint['img_channels']
        self.img_size = checkpoint['img_size']
        self.img_flat_size = self.img_channels * self.img_size * self.img_size
        self.hidden_dims = checkpoint['hidden_dims']
        self.num_diffusion_steps = checkpoint['num_diffusion_steps']

        # Get network architecture parameters from checkpoint if available
        activation = checkpoint.get('activation', 'relu')
        dropout_rate = checkpoint.get('dropout_rate', 0.0)

        # Re-initialize network with loaded parameters
        self.network = CNN_T(
            # MODIFICATION: Input dimension should match the modified __init__
            input_dim=self.img_channels, # Use img_channels only
            hidden_dims=self.hidden_dims,
            output_dim=self.img_channels,
            spatial_dim=self.img_size,
            activation=activation,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.network.load_state_dict(checkpoint['network'])
        # Re-initialize optimizer after loading network state
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        return self


def missdiff_impute(
    X0: torch.Tensor,
    n_epochs: int = 100,
    n_iter: int = 100,
    n_samples: int = 5,
    batch_size: int = 103,
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
    # do imputation 500 samples a time 
    X_imputed = torch.zeros_like(X0)
    for i in range(0, X0.shape[0], inference_batch_size):
        # Get the current batch
        batch_X0 = X0[i:i + inference_batch_size]
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