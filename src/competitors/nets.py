import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Union, Callable, Optional

# -----------------------------------------------
# Activation functions
# -----------------------------------------------
def get_activation(activation_name: str) -> Callable:
    """Returns activation function based on name."""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
        'identity': nn.Identity()
    }
    return activations.get(activation_name.lower(), nn.ReLU(inplace=True))

# -----------------------------------------------
# Residual Block for CNNNet
# -----------------------------------------------
class ResidualBlock(nn.Module):
    """Defines a residual block with two convolutional layers."""
    def __init__(self, hidden_dim: int, activation: str = 'relu'):
        super(ResidualBlock, self).__init__()
        self.act = get_activation(activation)
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self.act,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual  # Add the residual connection
        out = self.act(out)
        return out

# -----------------------------------------------
# Base MLP class (without time embedding)
# -----------------------------------------------
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [1024, 1024, 1024],
        activation: str = 'tanh',
        dropout_rate: float = 0.0,
    ):
        """
        Standard MLP network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            hidden_dims: List specifying the hidden dimensions for each layer
            activation: Activation function to use ('relu', 'tanh', 'silu', 'gelu', 'leaky_relu', 'identity')
            dropout_rate: Dropout probability (0 means no dropout)
        """
        super(MLP, self).__init__()
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(get_activation(activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# -----------------------------------------------
# MLP with Time Embedding
# -----------------------------------------------
class MLP_T(nn.Module):
    """Generic MLP with time embedding for diffusion and rectified flow models."""
    def __init__(self, input_dim, hidden_dims=[1024, 1024, 1024], output_dim=None, activation='tanh', dropout_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims 
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # Use Fourier embedding for time with 20 dimensions
        self.time_embed_dim = 20
        self.time_embed = FourierEmbedding(self.time_embed_dim)
        
        # Input projection with time
        self.input_proj = nn.Linear(input_dim + self.time_embed_dim, hidden_dims[0])
        
        # MLP backbone - structure fix: pass the full hidden_dims for proper layer creation
        self.mlp = MLP(
            input_dim=hidden_dims[0],
            output_dim=self.output_dim,
            hidden_dims=hidden_dims[1:] if len(hidden_dims) > 1 else [hidden_dims[0]],  # Skip the first layer which was handled by input_proj
            activation=activation,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x, t):
        """
        Generic forward pass handling any input and output dimensions.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            t: Time tensor of shape [batch_size, 1] or [batch_size]
            
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Process time embedding using Fourier embedding
        t_embed = self.time_embed(t.view(batch_size, 1))
        
        # Combine input and time embedding
        x_combined = torch.cat([x, t_embed], dim=1)
        
        # Project to hidden dimension
        x_proj = self.input_proj(x_combined)
        
        # Process through MLP
        return self.mlp(x_proj)

# -----------------------------------------------
# CNN Base class (without time embedding)
# -----------------------------------------------
class CNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spatial_dim: int,
        hidden_dims: List[int] = [64, 64, 64, 64],
        activation: str = 'relu',
        dropout_rate: float = 0.0,
    ):
        """
        Standard CNN network.
        
        Args:
            input_dim: Number of input channels
            output_dim: Number of output channels
            spatial_dim: Spatial dimension of the input (assuming square inputs)
            hidden_dims: List specifying the hidden channel dimensions for each residual block
            activation: Activation function to use ('relu', 'tanh', 'silu', 'gelu', 'leaky_relu', 'identity')
            dropout_rate: Dropout probability (0 means no dropout)
        """
        super(CNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spatial_dim = spatial_dim
        self.dropout_rate = dropout_rate
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            get_activation(activation),
        )
        
        # Residual blocks
        residual_layers = []
        for i in range(len(hidden_dims)):
            residual_layers.append(ResidualBlock(hidden_dims[i], activation))
            # Add dropout after each residual block if specified
            if dropout_rate > 0:
                residual_layers.append(nn.Dropout2d(dropout_rate))
                
        self.residual_layers = nn.Sequential(*residual_layers)
        
        # Final convolution
        self.final_conv = nn.Conv2d(hidden_dims[-1], output_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        out = self.initial_conv(x)
        out = self.residual_layers(out)
        out = self.final_conv(out)
        return out

# -----------------------------------------------
# CNN with Time Embedding
# -----------------------------------------------
class CNN_T(nn.Module):
    """Generic CNN with time embedding for diffusion and rectified flow models."""
    def __init__(self, input_dim, hidden_dims=[64, 64, 64, 64], output_dim=None, spatial_dim=32, activation='relu', dropout_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.spatial_dim = spatial_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # Modified CNN backbone that accepts an additional time channel
        self.cnn = CNN(
            input_dim=input_dim + 1,  # +1 for time channel
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            spatial_dim=spatial_dim,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x, t):
        """
        Generic forward pass for CNN with time embedding.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim, spatial_dim, spatial_dim]
            t: Time tensor of shape [batch_size, 1] or [batch_size]
            
        Returns:
            Tensor of shape [batch_size, output_dim, spatial_dim, spatial_dim]
        """
        batch_size = x.size(0)
        
        # Reshape t to [batch_size, 1, 1, 1]
        t = t.view(batch_size, 1, 1, 1)
        
        # Expand t to match spatial dimensions [batch_size, 1, spatial_dim, spatial_dim]
        t_expanded = t.expand(-1, 1, self.spatial_dim, self.spatial_dim)
        
        # Concatenate along channel dimension
        x_with_time = torch.cat([x, t_expanded], dim=1)
        
        # Process through CNN
        return self.cnn(x_with_time)

# -----------------------------------------------
# Fourier Embedding for Time Encoding
# -----------------------------------------------
class FourierEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Fourier feature embedding for time.
        
        Args:
            embed_dim: Dimension of the embedding (must be even)
        """
        super(FourierEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        # t: input time tensor of shape (batch_size, 1) or (batch_size,)
        t = t.view(-1, 1)  # Ensure shape is [batch_size, 1]
        
        # Create frequencies for embedding
        freqs = torch.arange(self.embed_dim // 2, dtype=torch.float32, device=t.device) * (2.0 * math.pi)
        
        # Reshape frequencies to [1, embed_dim//2]
        freqs = freqs.view(1, -1)
        
        # Calculate embeddings using broadcast: [batch_size, 1] * [1, embed_dim//2]
        t_cos = torch.cos(t * freqs)
        t_sin = torch.sin(t * freqs)
        
        # Combine sin and cos embeddings to get final shape [batch_size, embed_dim]
        embedding = torch.cat([t_cos, t_sin], dim=1)
        
        return embedding

# -----------------------------------------------
# RNN with Time Embedding for Time Series
# -----------------------------------------------
class RNN_T(nn.Module):
    """
    RNN-based network for time series data in Rectified Flow.
    Processes sequences and incorporates flow time t.
    Input z: (batch, seq_len, 2 * features) - Concatenated data and mask
    Input t: (batch, 1) - Flow time
    Output: (batch, seq_len, features) - Predicted velocity
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, rnn_type='LSTM', dropout=0.0):
        super().__init__()
        self.input_dim = input_dim # Should be 2 * num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # Should be num_features
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout

        # RNN layer (LSTM or GRU)
        # We add 1 to input_dim to accommodate the time t concatenated at each step
        rnn_input_dim = self.input_dim + 1 
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        else:
            raise ValueError("Unsupported RNN type. Choose 'LSTM' or 'GRU'.")

        # Output layer: Maps concatenated forward/backward hidden states to output dimension
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(hidden_dim * 2, output_dim) 

    def forward(self, z, t):
        # z shape: (batch, seq_len, 2 * features)
        # t shape: (batch, 1)
        
        batch_size, seq_len, _ = z.shape
        
        # Expand t to match the sequence length and feature dimension for concatenation
        # t shape becomes (batch, seq_len, 1)
        t_expanded = t.unsqueeze(1).expand(batch_size, seq_len, 1)
        
        # Concatenate t with z along the feature dimension
        # rnn_input shape: (batch, seq_len, 2 * features + 1)
        rnn_input = torch.cat((z, t_expanded), dim=-1)
        
        # Pass through RNN
        # rnn_output shape: (batch, seq_len, hidden_dim * 2) because bidirectional=True
        # h_n/c_n shape depends on layers and directionality, not directly used here for output
        rnn_output, _ = self.rnn(rnn_input) 
        
        # Apply the final fully connected layer to each time step's output
        # output shape: (batch, seq_len, features)
        output = self.fc(rnn_output)
        
        return output