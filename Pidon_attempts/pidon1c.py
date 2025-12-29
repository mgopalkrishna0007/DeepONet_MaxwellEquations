import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------------
# 3D U-Net Building Blocks
# -------------------------------

class Conv3DBlock(nn.Module):
    """3D Convolution block with residual connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.gelu = nn.GELU()
        
        # Residual connection
        self.residual = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        return x + residual


class Encoder(nn.Module):
    """Encoder block with conv + pooling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = Conv3DBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv_block(x)
        return self.pool(x), x  # Return pooled output and skip connection


class Decoder(nn.Module):
    """Decoder block with transposed conv + conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv_block = Conv3DBlock(out_ch * 2, out_ch)  # *2 for skip connection
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# -------------------------------
# Trunk Network (DeepONet-style)
# -------------------------------

class TrunkNetwork(nn.Module):
    """Processes coordinate information (x, y, z)"""
    def __init__(self, base_ch=32):
        super().__init__()
        # Input: 3 channels (x, y, z)
        self.enc1 = nn.Sequential(
            nn.Conv3d(3, base_ch, 3, padding=1),
            nn.GELU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, 3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(base_ch * 4, base_ch * 8, 3, padding=1),
            nn.GELU(),
            nn.MaxPool3d(2)
        )
    
    def forward(self, coords):
        f1 = self.enc1(coords)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        return f1, f2, f3, f4


# -------------------------------
# Deep Curl Operator Network
# -------------------------------

class DeepCurlOperator(nn.Module):
    """3D U-Net with DeepONet-style trunk network"""
    def __init__(self, base_ch=32):
        super().__init__()
        
        # Trunk network for coordinates
        self.trunk = TrunkNetwork(base_ch)
        
        # Branch network (U-Net) for field values
        # Input: 3 channels (Ex, Ey, Ez)
        self.enc1 = Encoder(3, base_ch)
        self.enc2 = Encoder(base_ch, base_ch * 2)
        self.enc3 = Encoder(base_ch * 2, base_ch * 4)
        self.enc4 = Encoder(base_ch * 4, base_ch * 8)
        
        # Bottleneck - combines trunk and branch
        self.bottleneck = Conv3DBlock(base_ch * 8 + base_ch * 8, base_ch * 8)
        
        # Decoder
        self.dec4 = Decoder(base_ch * 8, base_ch * 4)
        self.dec3 = Decoder(base_ch * 4, base_ch * 2)
        self.dec2 = Decoder(base_ch * 2, base_ch)
        self.dec1 = Decoder(base_ch, base_ch)
        
        # Output layer: 3 channels (curl_x, curl_y, curl_z)
        self.out_conv = nn.Conv3d(base_ch, 3, kernel_size=1)
    
    def forward(self, coords, fields):
        """
        Args:
            coords: [B, 3, X, Y, Z] - spatial coordinates
            fields: [B, 3, X, Y, Z] - electric field components
        Returns:
            curl: [B, 3, X, Y, Z] - curl of electric field
        """
        # Trunk network (coordinate processing)
        trunk_f1, trunk_f2, trunk_f3, trunk_f4 = self.trunk(coords)
        
        # Branch network (field processing)
        x, skip1 = self.enc1(fields)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Combine trunk and branch at bottleneck
        x = torch.cat([x, trunk_f4], dim=1)
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        curl = self.out_conv(x)
        return curl


# -------------------------------
# Dataset Class
# -------------------------------

class EMFieldDataset(Dataset):
    """Dataset for electromagnetic fields and their curls"""
    def __init__(self, E_data, curl_data, grid_coords):
        """
        Args:
            E_data: [N, 3, 32, 32, 32]
            curl_data: [N, 3, 32, 32, 32]
            grid_coords: [3, 32, 32, 32] - normalized coordinates
        """
        self.E_data = torch.FloatTensor(E_data)
        self.curl_data = torch.FloatTensor(curl_data)
        self.coords = torch.FloatTensor(grid_coords)
    
    def __len__(self):
        return len(self.E_data)
    
    def __getitem__(self, idx):
        return self.coords, self.E_data[idx], self.curl_data[idx]


# -------------------------------
# Training Functions
# -------------------------------

def normalize_curl(curl):
    """Component-wise normalization"""
    batch_size = curl.shape[0]
    normalized = torch.zeros_like(curl)
    
    for b in range(batch_size):
        for c in range(3):
            max_val = torch.abs(curl[b, c]).max()
            if max_val > 1e-8:
                normalized[b, c] = curl[b, c] / max_val
            else:
                normalized[b, c] = curl[b, c]
    
    return normalized


def mean_relative_error(pred, target):
    """Compute Mean Relative Error"""
    abs_error = torch.abs(pred - target)
    rel_error = abs_error / (torch.abs(target) + 1e-8)
    return rel_error.mean().item()


def train_model(model, train_loader, test_loader, epochs=1000, lr=1e-4, device='cuda'):
    """Train the Deep Curl Operator"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_mres = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for coords, E_fields, curl_true in train_loader:
            coords = coords.to(device)
            E_fields = E_fields.to(device)
            curl_true = curl_true.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            curl_pred = model(coords, E_fields)
            
            # Normalize for training
            curl_pred_norm = normalize_curl(curl_pred)
            curl_true_norm = normalize_curl(curl_true)
            
            # Compute loss
            loss = criterion(curl_pred_norm, curl_true_norm)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation
        if (epoch + 1) % 50 == 0:
            model.eval()
            test_mre = 0.0
            
            with torch.no_grad():
                for coords, E_fields, curl_true in test_loader:
                    coords = coords.to(device)
                    E_fields = E_fields.to(device)
                    curl_true = curl_true.to(device)
                    
                    curl_pred = model(coords, E_fields)
                    test_mre += mean_relative_error(curl_pred, curl_true)
            
            test_mre /= len(test_loader)
            test_mres.append(test_mre)
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Test MRE: {test_mre:.6f}")
    
    return train_losses, test_mres


# -------------------------------
# Visualization Functions
# -------------------------------

def visualize_results(model, coords, E_field, curl_true, slice_idx=16, device='cuda'):
    """Visualize predictions vs ground truth"""
    model.eval()
    
    with torch.no_grad():
        coords_tensor = torch.FloatTensor(coords).unsqueeze(0).to(device)
        E_tensor = torch.FloatTensor(E_field).unsqueeze(0).to(device)
        curl_pred = model(coords_tensor, E_tensor).cpu().numpy()[0]
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    components = ['x', 'y', 'z']
    
    for i, comp in enumerate(components):
        # Input E-field
        im0 = axes[i, 0].imshow(E_field[i, :, :, slice_idx], cmap='RdBu_r')
        axes[i, 0].set_title(f'Input E_{comp}')
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Ground truth curl
        vmax = np.abs(curl_true[i]).max()
        im1 = axes[i, 1].imshow(curl_true[i, :, :, slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f'GT Curl_{comp}')
        plt.colorbar(im1, ax=axes[i, 1])
        
        # Predicted curl
        im2 = axes[i, 2].imshow(curl_pred[i, :, :, slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 2].set_title(f'Pred Curl_{comp}')
        plt.colorbar(im2, ax=axes[i, 2])
        
        # Absolute error
        abs_err = np.abs(curl_pred[i] - curl_true[i])
        im3 = axes[i, 3].imshow(abs_err[:, :, slice_idx], cmap='hot')
        axes[i, 3].set_title(f'Abs Error Curl_{comp}')
        plt.colorbar(im3, ax=axes[i, 3])
        
        # Relative error
        rel_err = abs_err / (np.abs(curl_true[i]) + 1e-8)
        im4 = axes[i, 4].imshow(rel_err[:, :, slice_idx], cmap='hot', vmin=0, vmax=1)
        axes[i, 4].set_title(f'Rel Error Curl_{comp}')
        plt.colorbar(im4, ax=axes[i, 4])
    
    plt.tight_layout()
    return fig


# -------------------------------
# Main Training Script
# -------------------------------

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load your data (assuming E_train and curl_train are already generated)
    # E_train, curl_train = generate_dataset(1000)  # Use your existing function
    
    # Create normalized coordinate grid
    Nx = Ny = Nz = 32
    L = 19.2e-3
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    z = np.linspace(0, L, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Normalize coordinates to [0, 1]
    coords = np.stack([X/L, Y/L, Z/L], axis=0).astype(np.float32)
    
    # Split dataset (80/20)
    # train_size = int(0.8 * len(E_train))
    # E_train_split = E_train[:train_size]
    # curl_train_split = curl_train[:train_size]
    # E_test = E_train[train_size:]
    # curl_test = curl_train[train_size:]
    
    # Create datasets and loaders
    # train_dataset = EMFieldDataset(E_train_split, curl_train_split, coords)
    # test_dataset = EMFieldDataset(E_test, curl_test, coords)
    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = DeepCurlOperator(base_ch=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    # train_losses, test_mres = train_model(
    #     model, train_loader, test_loader,
    #     epochs=1000, lr=1e-4, device=device
    # )
    
    # Save model
    # torch.save(model.state_dict(), 'deep_curl_operator.pth')
    
    print("Training complete!")


if __name__ == "__main__":
    main()