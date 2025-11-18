import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Defines a residual block with two convolutional layers."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual  # Add the residual connection
        out = self.relu(out)
        return out

class CNNNet(nn.Module):
    def __init__(self, dimvec):
        super(CNNNet, self).__init__()
        self.d = int((dimvec/3) ** 0.5)
        # Update the input layer to accept 8 input channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1, bias=False), # 3 + 3 + 3 + 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stack multiple residual blocks
        self.residual_layers = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        # Update the final convolution layer to output 3 channels
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        m = x[:, (self.d * self.d * 6):]
        batch_size = x.size(0)
        x = x[:, : (self.d * self.d * 9)]  # Now we have 9 input channels 3 + 3 + 3
        x = x.view(batch_size, 9, self.d, self.d)  # Reshape for 9 channels
        t = t.view(batch_size, 1, 1, 1)
        t_expanded = t.expand(-1, 1, self.d, self.d)
        x = torch.cat([x, t_expanded], dim=1)  # Now x has 10 channels
        out = self.initial_conv(x)
        out = self.residual_layers(out)
        out = self.final_conv(out)
        out = out.view(batch_size, -1)
        
        # repeat m 3 times to match the number of output channels
        # m = m.repeat(1, 3)
        return (1 - m) * out
    
class CNNNet2(nn.Module):
    def __init__(self, dimvec):
        super(CNNNet2, self).__init__()
        self.d = int((dimvec/3) ** 0.5)
        # Update the input layer to accept 8 input channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stack multiple residual blocks
        self.residual_layers = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        # Update the final convolution layer to output 3 channels
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, self.d, self.d)  # Reshape for 3 channels
        out = self.initial_conv(x)
        out = self.residual_layers(out)
        out = self.final_conv(out)
        out = out.view(batch_size, -1)
        
        return out
    
class CNNNet2_T(nn.Module):
    def __init__(self, dimvec):
        super(CNNNet2_T, self).__init__()
        self.d = int((dimvec/3) ** 0.5)
        # Update the input layer to accept 4 input channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stack multiple residual blocks
        self.residual_layers = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        # Update the final convolution layer to output 3 channels
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, self.d, self.d)  # Reshape for 3 channels
        t = t.view(batch_size, 1, 1, 1)
        t_expanded = t.expand(-1, 1, self.d, self.d)
        x = torch.cat([x, t_expanded], dim=1)  # Now x has 4 channels
        out = self.initial_conv(x)
        out = self.residual_layers(out)
        out = self.final_conv(out)
        out = out.view(batch_size, -1)
        
        return out
    
    
class CNNNet3(nn.Module):
    def __init__(self, dimvec):
        super(CNNNet3, self).__init__()
        self.d = int((dimvec/3) ** 0.5)
        # Update the input layer to accept 8 input channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False), # 3 + 3 + 3 + 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stack multiple residual blocks
        self.residual_layers = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        # Update the final convolution layer to output 3 channels
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        batch_size = x.shape[0]
        t = t.view(batch_size, 1, 1, 1)
        t_expanded = t.expand(-1, 1, self.d, self.d)
        x = torch.cat([x, t_expanded], dim=1)  # Now x has 10 channels
        out = self.initial_conv(x)
        out = self.residual_layers(out)
        out = self.final_conv(out)
        
        return out

# Test function remains largely the same, but we need to adjust for the new number of input channels

def test_improved_cnn_net():
    d = 32  # Example spatial dimension (height and width)
    batch_size = 100  # Example batch size

    # Instantiate the model
    model = CNNNet(d*d*3)

    # Create dummy input data
    # The input x should have a total size of [batch_size, 9 * d * d]
    x_data_size = 9 * d * d  # Total size of x, now with 9 input channels
    x = torch.randn(batch_size, x_data_size)

    # Create dummy time input t
    t = torch.randn(batch_size, 1)

    # Run the model
    output = model(x, t)

    # Print the output shape
    print('Output shape:', output.shape)
    # Optionally, print the output values
    print('Output values:', output)

# Run the test function
test_improved_cnn_net()
