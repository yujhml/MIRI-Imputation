import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d, hidden_dims=[1001, 1001, 1001], dropout_prob=0.2):
        super(MLP, self).__init__()
        input_dim = 3 * d + 1
        self.d = d
        self.flatten = nn.Flatten()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            # layers.append(nn.LayerNorm(hdim))
            layers.append(nn.SiLU())  # SiLU = Swish
            # layers.append(nn.Dropout(dropout_prob))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, d))  # Final output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        m = x[:, -self.d:]
        x = self.flatten(torch.cat([x, t], dim=1))
        return (1-m)*self.mlp(x)
    
    
class MLP2(nn.Module):
    def __init__(self, d, hidden_dims=[1001, 1001, 1001], dropout_prob=0.2):
        super(MLP2, self).__init__()
        input_dim = 1 * d + 1
        self.flatten = nn.Flatten()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            # layers.append(nn.LayerNorm(hdim))
            layers.append(nn.SiLU())  # SiLU = Swish
            # layers.append(nn.Dropout(dropout_prob))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, d))  # Final output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x = self.flatten(torch.cat([x, t], dim=1))
        return self.mlp(x)