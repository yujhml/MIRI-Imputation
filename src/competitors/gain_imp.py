import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

from src.competitors.nets import CNN

EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------
# GainModel using CNN from models.nets for both generator and discriminator.
class GainModel(nn.Module):
    def __init__(self, input_dim: int, loss_alpha: float = 10, spatial_dim: int = 32,
                 hidden_dims: List[int] = [64, 64, 64, 64],   
                 activation: str = 'relu',   
                 dropout_rate: float = 0.0):   
        """
        Args:
            input_dim (int): Flattened input dimension (c*d*d).
            loss_alpha (float): Weight for MSE loss component.
            spatial_dim (int): Image height/width (assumed square).
            hidden_dims (List[int]): list of hidden dimensions for CNN layers.
            activation (str): Activation function for CNN.
            dropout_rate (float): Dropout probability for CNN.
        """
        super(GainModel, self).__init__()
        self.spatial_dim = spatial_dim
        self.channels = input_dim // (spatial_dim * spatial_dim)  # infer number of channels
        assert self.channels in [1, 3], "Only 1-channel and 3-channel images are supported."
        
        # Generator network: CNN operating on concatenated (image, mask)
        self.generator_net = CNN(input_dim=2 * self.channels,
                                 output_dim=self.channels,
                                 spatial_dim=spatial_dim,
                                 hidden_dims=hidden_dims,   
                                 activation=activation,   
                                 dropout_rate=dropout_rate)   
        self.generator_fc = nn.Linear(self.channels * spatial_dim * spatial_dim,
                                      self.channels * spatial_dim * spatial_dim)
        self.generator_sigmoid = nn.Sigmoid()
        nn.init.xavier_normal_(self.generator_fc.weight)
        
        # Discriminator network: same structure.
        self.discriminator_net = CNN(input_dim=2 * self.channels,
                                     output_dim=self.channels,
                                     spatial_dim=spatial_dim,
                                     hidden_dims=hidden_dims,   
                                     activation=activation,   
                                     dropout_rate=dropout_rate)   
        self.discriminator_fc = nn.Linear(self.channels * spatial_dim * spatial_dim,
                                          self.channels * spatial_dim * spatial_dim)
        self.discriminator_sigmoid = nn.Sigmoid()
        nn.init.xavier_normal_(self.discriminator_fc.weight)
        
        self.loss_alpha = loss_alpha

    def generator(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Flattened image tensor of shape [batch, c*d*d]
            mask: Flattened mask tensor of same shape (values 0 or 1)
        Returns:
            Generator output tensor (flattened).
        """
        batch_size = X.size(0)
        X = X.view(batch_size, self.channels, self.spatial_dim, self.spatial_dim)
        mask = mask.view(batch_size, self.channels, self.spatial_dim, self.spatial_dim)
        inputs = torch.cat([X, mask], dim=1)  # shape: (batch, 2*c, d, d)
        gen_out = self.generator_net(inputs)
        return self.generator_sigmoid(self.generator_fc(gen_out.view(batch_size, -1)))

    def discriminator(self, X: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Flattened image tensor (either original or imputed), shape [batch, c*d*d]
            hints: Flattened hints tensor, shape [batch, c*d*d]
        Returns:
            Discriminator probability output.
        """
        batch_size = X.size(0)
        X = X.view(batch_size, self.channels, self.spatial_dim, self.spatial_dim)
        hints = hints.view(batch_size, self.channels, self.spatial_dim, self.spatial_dim)
        inputs = torch.cat([X, hints], dim=1)
        disc_out = self.discriminator_net(inputs)
        return self.discriminator_sigmoid(self.discriminator_fc(disc_out.view(batch_size, -1)))
    
    def discr_loss(self, X: torch.Tensor, M: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)
        G_sample = self.generator(X, M)  # Generated values
        X = X.view(batch_size, -1)
        M = M.view(batch_size, -1)
        # Combine observed and generated values.
        X_hat = X * M + G_sample * (1 - M)
        D_prob = self.discriminator(X_hat, H)
        return -torch.mean(M * torch.log(D_prob + EPS) + (1 - M) * torch.log(1.0 - D_prob + EPS))

    def gen_loss(self, X: torch.Tensor, M: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        batch_size = X.size(0)
        G_sample = self.generator(X, M)
        X = X.view(batch_size, -1)
        M = M.view(batch_size, -1)
        X_hat = X * M + G_sample * (1 - M)
        D_prob = self.discriminator(X_hat, H)
        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + EPS))
        MSE_train_loss = torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)
        return G_loss1 + self.loss_alpha * MSE_train_loss

# -----------------------------------------------
# Wrapper class for GAIN Imputation (using the above GainModel)
class GainImputation:
    def __init__(self, batch_size: int = 64, n_epochs: int = 100, hint_rate: float = 0.9, loss_alpha: float = 10,
                 spatial_dim: int = 32,
                 hidden_dims: List[int] = [64, 64, 64, 64],   
                 activation: str = 'relu',   
                 dropout_rate: float = 0.0,
                 learning_rate: float = 0.001):   
        # Reduce default batch size from 256 to 64
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hint_rate = hint_rate  # Fix: use the parameter, not self.hint_rate
        self.loss_alpha = loss_alpha
        self.model = None
        self.spatial_dim = spatial_dim
        self.hidden_dims = hidden_dims   
        self.activation = activation   
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def fit(self, X: torch.Tensor) -> "GainImputation":
        X = X.clone().to(DEVICE)
        
        # Handle 4D input - reshape to 2D for internal processing
        if len(X.shape) == 4:
            N, C, H, W = X.shape
            X = X.view(N, C*H*W)
        
        no, dim = X.shape
        X_norm = X.clone().cpu()
        mask = 1 - torch.isnan(X).float().to(DEVICE)
        X_norm = torch.nan_to_num(X_norm).to(DEVICE)
        self.model = GainModel(dim, loss_alpha=self.loss_alpha, spatial_dim=self.spatial_dim,
                               hidden_dims=self.hidden_dims,   
                               activation=self.activation,   
                               dropout_rate=self.dropout_rate).to(DEVICE)
        D_solver = optim.Adam(
            list(self.model.discriminator_net.parameters()) + list(self.model.discriminator_fc.parameters()),
            lr=self.learning_rate
        )
        G_solver = optim.Adam(
            list(self.model.generator_net.parameters()) + list(self.model.generator_fc.parameters()),
            lr=self.learning_rate
        )
        
        def sample_idx(m: int, n: int) -> np.ndarray:
            return np.random.permutation(m)[:n]
        
        def sample_Z(m: int, n: int) -> torch.Tensor:
            return torch.from_numpy(np.random.uniform(0.0, 0.01, size=[m, n])).float().to(DEVICE)
        
        def sample_M(m: int, n: int, p: float) -> torch.Tensor:
            unif_prob = np.random.uniform(0.0, 1.0, size=[m, n])
            M_arr = (unif_prob > p).astype(float)
            return torch.from_numpy(M_arr).float().to(DEVICE)
        
        def sample():
            mb_size = min(self.batch_size, no)
            mb_idx = sample_idx(no, mb_size)
            x_mb = X_norm[mb_idx, :].clone()
            m_mb = mask[mb_idx, :].clone()
            z_mb = sample_Z(mb_size, dim)
            h_mb = sample_M(mb_size, dim, 1 - self.hint_rate)
            h_mb = m_mb * h_mb
            x_mb = m_mb * x_mb + (1 - m_mb) * z_mb
            return x_mb, h_mb, m_mb
        
        for it in range(self.n_epochs):
            # Memory optimization: Periodically clear CUDA cache
            if it > 0 and it % 50 == 0:
                torch.cuda.empty_cache()
                
            # Zero the entire model gradients at the start of each iteration.
            self.model.zero_grad()
            
            # Update discriminator.
            D_solver.zero_grad()
            x_mb, h_mb, m_mb = sample()
            D_loss = self.model.discr_loss(x_mb, m_mb, h_mb)
            D_loss.backward()
            D_solver.step()
            
            # Store discriminator loss value
            D_loss_val = D_loss.item()
            
            # Memory optimization: Delete intermediate tensors
            del x_mb, h_mb, m_mb, D_loss
            
            # Update generator.
            G_solver.zero_grad()
            x_mb, h_mb, m_mb = sample()
            G_loss = self.model.gen_loss(x_mb, m_mb, h_mb)
            # Explicitly use default retain_graph=False to ensure the graph is freed.
            G_loss.backward(retain_graph=False)
            G_solver.step()
            
            # Store generator loss value
            G_loss_val = G_loss.item()
            
            # Print losses every 100 epochs
            if it % 100 == 0:
                print(f"Epoch {it}/{self.n_epochs}: D_loss = {D_loss_val:.4f}, G_loss = {G_loss_val:.4f}")
            
            # Memory optimization: Delete intermediate tensors
            del x_mb, h_mb, m_mb, G_loss
        
        return self

    def transform(self, Xmiss: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
            
        Xmiss = Xmiss.clone().to(DEVICE)
        
        # Store original shape for 4D inputs
        original_shape = None
        if len(Xmiss.shape) == 4:
            original_shape = Xmiss.shape
            N, C, H, W = original_shape
            Xmiss = Xmiss.view(N, C*H*W)
        
        X = Xmiss
        no, dim = X.shape
        mask = 1 - torch.isnan(X).float().to(DEVICE)
        X = torch.nan_to_num(X).to(DEVICE)
        z = torch.from_numpy(np.random.uniform(0.0, 0.01, size=[no, dim])).float().to(DEVICE)
        X_in = mask * X + (1 - mask) * z
        
        # Process all data at once, with gradient tracking disabled
        with torch.no_grad():
            imputed = self.model.generator(X_in, mask)
        
        result = (mask * X + (1 - mask) * imputed).detach()
        
        # Reshape back to original shape if it was 4D
        if original_shape is not None:
            result = result.view(original_shape)
            
        return result

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

# -----------------------------------------------
# GAIN imputation function
def gain_impute(X0: torch.Tensor,
                batch_size: int = 64,  # Reduced from 256 to 64
                n_epochs: int = 1000, 
                hint_rate: float = 0.9,
                loss_alpha: float = 10,
                hidden_dims: List[int] = [64, 64, 64, 64],   
                activation: str = 'relu',   
                dropout_rate: float = 0.0,
                learning_rate: float = 0.001):   
    # Check if input is a 4D tensor (N, C, H, W)
    if len(X0.shape) != 4:
        raise ValueError("Input tensor must be 4D (N, C, H, W).")
    
    # Extract spatial dimension from the input tensor
    spatial_dim = X0.shape[2]  # Assuming square images where H=W

    # Move tensors to device.
    X0 = X0.to(DEVICE)
    
    # Get missing mask M where missing entries are labeled as 0.
    M = (~torch.isnan(X0)).float().to(DEVICE)
    
    # For GAIN, mark missing entries as NaN.
    X0_impute = X0.clone()
    
    # Train GAIN model once with more epochs.
    print(f"Training GAIN model with {n_epochs} epochs...")
    gain_model = GainImputation(batch_size=batch_size,
                                n_epochs=n_epochs,
                                hint_rate=hint_rate,
                                loss_alpha=loss_alpha,
                                spatial_dim=spatial_dim,
                                hidden_dims=hidden_dims,   
                                activation=activation,   
                                dropout_rate=dropout_rate,
                                learning_rate=learning_rate)   
    
    imputed = gain_model.fit_transform(X0_impute)
    
    # Memory optimization: Free up intermediate tensors
    gain_model = None
    torch.cuda.empty_cache()
    
    # Update missing entries.
    X0_impute[M == 0] = imputed[M == 0]
    
    return X0_impute