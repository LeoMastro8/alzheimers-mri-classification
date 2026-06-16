import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    """
    A single 3D Convolutional Block: Conv3d -> BatchNorm -> ReLU -> MaxPool
    """
    def __init__(self, in_channels, out_channels, pool=True):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class EarlyFusion3DCNN(nn.Module):
    """
    Custom 3D CNN for ADNI multi-class classification.
    Expects input shape: [B, 3, 128, 128, 128]
    """
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(EarlyFusion3DCNN, self).__init__()
        
        # --- Feature Extractor ---
        self.block1 = ConvBlock3D(in_channels=3, out_channels=32)
        self.block2 = ConvBlock3D(in_channels=32, out_channels=64)
        self.block3 = ConvBlock3D(in_channels=64, out_channels=128)
        
        # Block 4: MaxPool is omitted here because we use AdaptiveAvgPool right after
        self.block4 = ConvBlock3D(in_channels=128, out_channels=256, pool=False)
        
        # Squeeze the spatial dimensions to 1x1x1
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # --- Classification Head ---
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
            # NO SOFTMAX HERE! Output raw logits for nn.CrossEntropyLoss
        )

    def forward(self, x):
        # Pass through the 4 convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Pool and classify
        x = self.global_pool(x)
        out = self.head(x)
        
        return out

# ==========================================
# Testing and Parameter Counting
# ==========================================
if __name__ == "__main__":
    # Initialize the model
    model = EarlyFusion3DCNN(num_classes=3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Initialized Successfully!")
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # Create a dummy input tensor: [Batch Size, Channels, Depth, Height, Width]
    # Simulating a batch of 2 subjects, with 3 early-fused modalities (128x128x128)
    dummy_input = torch.randn(2, 3, 128, 128, 128)
    
    print(f"\nPassing dummy input of shape: {dummy_input.shape}...")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output shape (Expected [2, 3]): {output.shape}")
    print(f"Sample Logits: \n{output.detach().numpy()}")