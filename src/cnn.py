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
        self.d = int(dimvec ** 0.5)
        # Update the input layer to accept 5 input channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False), # 1 + 1 + 1 + 1
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
        # Update the final convolution layer to output 1 channels
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        m = x[:, (self.d * self.d * 2):]
        batch_size = x.size(0)
        x = x[:, : (self.d * self.d * 3)]  # Now we have 7 input channels 1 + 1 + 1
        x = x.view(batch_size, 3, self.d, self.d)  # Reshape for 3 channels
        t = t.view(batch_size, 1, 1, 1)
        t_expanded = t.expand(-1, 1, self.d, self.d)
        x = torch.cat([x, t_expanded], dim=1)  # Now x has 4 channels
        out = self.initial_conv(x)
        out = self.residual_layers(out)
        out = self.final_conv(out)
        out = out.view(batch_size, -1)
        
        # repeat m 3 times to match the number of output channels
        # m = m.repeat(1, 3)
        return (1 - m) * out

# Test function remains largely the same, but we need to adjust for the new number of input channels

def test_improved_cnn_net():
    d = 32  # Example spatial dimension (height and width)
    batch_size = 100  # Example batch size

    # Instantiate the model
    model = CNNNet(d*d)

    # Create dummy input data
    # The input x should have a total size of [batch_size, 3 * d * d]
    x_data_size = 3 * d * d  # Total size of x, now with 3 input channels
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
