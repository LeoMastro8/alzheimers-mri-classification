import torch
import torch.nn as nn
from monai.networks.nets import resnet18

class MedicalNetTransfer(nn.Module):
    def __init__(self, pretrained_weights_path=None):
        super(MedicalNetTransfer, self).__init__()
        
        # MedicalNet specifically requires shortcut_type='A'
        self.backbone = resnet18(
            spatial_dims=3, 
            n_input_channels=1, 
            num_classes=3, 
            shortcut_type='A' 
        )
        
        # Load the downloaded MedicalNet weights into the body
        if pretrained_weights_path:
            print(f"Loading pre-trained MedicalNet weights from {pretrained_weights_path}...")
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            
            # Clean dictionary keys (remove 'module.' if it exists)
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict['state_dict'].items()}
            
            # Load weights safely (strict=False ignores the final 3-class layer mismatch)
            self.backbone.load_state_dict(clean_state_dict, strict=False)
            print("Weights loaded successfully!")
            
        # Modify Conv1 to accept 3 channels (T1, T2, FLAIR) instead of 1
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv3d(
            in_channels=3, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=False
        )
        
        # Copy the pre-trained 1-channel weights across the 3 new channels
        with torch.no_grad():
            self.backbone.conv1.weight[:] = old_conv.weight.repeat(1, 3, 1, 1, 1) / 3.0
            
    def forward(self, x):
        return self.backbone(x)