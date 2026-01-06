"""
DCO-UNet 3D: Differentiable Curl Operator via Deep Learning
Physics-informed 3D convolutional neural network for learning curl operator from E-field to curl(E)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import json

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# -------------------------------
# Physical constants & Grid
# -------------------------------
c0 = 3e8  # Speed of light in vacuum (m/s)
pi = np.pi
Nx = Ny = Nz = 32  # Grid dimensions
L = 19.2e-3  # Domain length (m)
dx = dy = dz = L / Nx  # Grid spacing

# Create coordinate grid
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
z = np.linspace(0, L, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# -------------------------------
# Physics-Correct Finite Difference Curl
# -------------------------------
def compute_curl_finite_difference(E):
    """
    Compute curl using second-order central differences.
    Input: E of shape [3, Nx, Ny, Nz] or [B, 3, Nx, Ny, Nz]
    Output: curl(E) of same shape
    
    Physical curl definition:
    (∇×E)_x = ∂E_z/∂y - ∂E_y/∂z
    (∇×E)_y = ∂E_x/∂z - ∂E_z/∂x
    (∇×E)_z = ∂E_y/∂x - ∂E_x/∂y
    
    Central difference: ∂f/∂x|_i = (f_{i+1} - f_{i-1}) / (2*dx)
    Boundary: one-sided difference for stability
    """
    if E.ndim == 4:
        E = E[np.newaxis, ...]  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, _, Nx, Ny, Nz = E.shape
    curl = np.zeros_like(E)
    
    # Extract vector components
    Ex, Ey, Ez = E[:, 0], E[:, 1], E[:, 2]
    
    # ---------- ∂/∂x derivatives ----------
    # Using central difference with proper grid spacing
    dEy_dx = np.zeros_like(Ey)
    dEy_dx[:, 1:-1, :, :] = (Ey[:, 2:, :, :] - Ey[:, :-2, :, :]) / (2 * dx)  # Central
    dEy_dx[:, 0, :, :] = (Ey[:, 1, :, :] - Ey[:, 0, :, :]) / dx  # Forward at boundary
    dEy_dx[:, -1, :, :] = (Ey[:, -1, :, :] - Ey[:, -2, :, :]) / dx  # Backward at boundary
    
    dEz_dx = np.zeros_like(Ez)
    dEz_dx[:, 1:-1, :, :] = (Ez[:, 2:, :, :] - Ez[:, :-2, :, :]) / (2 * dx)  # Central
    dEz_dx[:, 0, :, :] = (Ez[:, 1, :, :] - Ez[:, 0, :, :]) / dx  # Forward
    dEz_dx[:, -1, :, :] = (Ez[:, -1, :, :] - Ez[:, -2, :, :]) / dx  # Backward
    
    # ---------- ∂/∂y derivatives ----------
    dEx_dy = np.zeros_like(Ex)
    dEx_dy[:, :, 1:-1, :] = (Ex[:, :, 2:, :] - Ex[:, :, :-2, :]) / (2 * dy)  # Central
    dEx_dy[:, :, 0, :] = (Ex[:, :, 1, :] - Ex[:, :, 0, :]) / dy  # Forward
    dEx_dy[:, :, -1, :] = (Ex[:, :, -1, :] - Ex[:, :, -2, :]) / dy  # Backward
    
    dEz_dy = np.zeros_like(Ez)
    dEz_dy[:, :, 1:-1, :] = (Ez[:, :, 2:, :] - Ez[:, :, :-2, :]) / (2 * dy)  # Central
    dEz_dy[:, :, 0, :] = (Ez[:, :, 1, :] - Ez[:, :, 0, :]) / dy  # Forward
    dEz_dy[:, :, -1, :] = (Ez[:, :, -1, :] - Ez[:, :, -2, :]) / dy  # Backward
    
    # ---------- ∂/∂z derivatives ----------
    dEx_dz = np.zeros_like(Ex)
    dEx_dz[:, :, :, 1:-1] = (Ex[:, :, :, 2:] - Ex[:, :, :, :-2]) / (2 * dz)  # Central
    dEx_dz[:, :, :, 0] = (Ex[:, :, :, 1] - Ex[:, :, :, 0]) / dz  # Forward
    dEx_dz[:, :, :, -1] = (Ex[:, :, :, -1] - Ex[:, :, :, -2]) / dz  # Backward
    
    dEy_dz = np.zeros_like(Ey)
    dEy_dz[:, :, :, 1:-1] = (Ey[:, :, :, 2:] - Ey[:, :, :, :-2]) / (2 * dz)  # Central
    dEy_dz[:, :, :, 0] = (Ey[:, :, :, 1] - Ey[:, :, :, 0]) / dz  # Forward
    dEy_dz[:, :, :, -1] = (Ey[:, :, :, -1] - Ey[:, :, :, -2]) / dz  # Backward
    
    # ---------- Compute curl components ----------
    curl[:, 0] = dEz_dy - dEy_dz  # (∇×E)_x = ∂Ez/∂y - ∂Ey/∂z
    curl[:, 1] = dEx_dz - dEz_dx  # (∇×E)_y = ∂Ex/∂z - ∂Ez/∂x
    curl[:, 2] = dEy_dx - dEx_dy  # (∇×E)_z = ∂Ey/∂x - ∂Ex/∂y
    
    if squeeze_output:
        curl = curl[0]  # Remove batch dimension
    
    return curl

# -------------------------------
# Plane Wave Generator with Mixed Phase
# -------------------------------
# def generate_plane_wave():
#     """
#     Generate a random plane wave E-field and its curl.
    
#     Key improvements:
#     1. Mixed phase (cosine - sine) to resolve sign ambiguity
#     2. Normalized coordinates as part of input
#     3. Proper curl computation with grid spacing
#     """
#     # Random wave parameters
#     theta = np.random.uniform(0, pi)  # Polar angle
#     phi = np.random.uniform(0, 2 * pi)  # Azimuthal angle
#     f = np.random.uniform(1e6, 50e9)  # Frequency 1MHz - 50GHz
#     k0 = 2 * pi * f / c0  # Wavenumber
#     kx = k0 * np.sin(theta) * np.cos(phi)
#     ky = k0 * np.sin(theta) * np.sin(phi)
#     kz = k0 * np.cos(theta)
    
#     # Random polarization vector (satisfying ∇·E = 0)
#     Ex0 = np.random.randn()
#     Ey0 = np.random.randn()
#     Ez0 = -(kx * Ex0 + ky * Ey0) / kz if abs(kz) > 1e-6 else 0.0
#     E0 = np.array([Ex0, Ey0, Ez0])
    
#     # Phase term
#     phase = kx * X + ky * Y + kz * Z
    
#     # ---------- KEY FIX: Mixed phase to resolve sign ambiguity ----------
#     # Using cosine - sine instead of just cosine to avoid sign symmetry
#     Ex = E0[0] * (np.cos(phase) - np.sin(phase))
#     Ey = E0[1] * (np.cos(phase) - np.sin(phase))
#     Ez = E0[2] * (np.cos(phase) - np.sin(phase))
    
#     # Stack into vector field
#     E = np.stack([Ex, Ey, Ez], axis=0)
    
#     # Compute curl using finite differences
#     curlE = compute_curl_finite_difference(E)
    
#     return E.astype(np.float32), curlE.astype(np.float32)

def generate_single_plane_wave():
    """
    Generate a SINGLE random plane wave E-field.
    Same as original generate_plane_wave() but returns only E, no curl.
    """
    # Random wave parameters
    theta = np.random.uniform(0, pi)  # Polar angle
    phi = np.random.uniform(0, 2 * pi)  # Azimuthal angle
    f = np.random.uniform(1e6, 50e9)  # Frequency 1MHz - 50GHz
    k0 = 2 * pi * f / c0  # Wavenumber
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)
    
    # Random polarization vector (satisfying ∇·E = 0)
    Ex0 = np.random.randn()
    Ey0 = np.random.randn()
    Ez0 = -(kx * Ex0 + ky * Ey0) / kz if abs(kz) > 1e-6 else 0.0
    E0 = np.array([Ex0, Ey0, Ez0])
    
    # Phase term
    phase = kx * X + ky * Y + kz * Z
    
    # Mixed phase to resolve sign ambiguity
    Ex = E0[0] * (np.cos(phase) - np.sin(phase))
    Ey = E0[1] * (np.cos(phase) - np.sin(phase))
    Ez = E0[2] * (np.cos(phase) - np.sin(phase))
    
    # Stack into vector field
    E = np.stack([Ex, Ey, Ez], axis=0)
    
    return E.astype(np.float32)

def generate_plane_wave_superposition(num_waves=3):
    """
    Generate superposition of multiple plane waves E-field and its curl.
    Matches paper's equation: E = Σ E₀,i exp{-j k_i [...]}
    
    Args:
        num_waves: Fixed number of plane waves to superpose (default=3)
    
    Returns:
        E: Superposed E-field [3, Nx, Ny, Nz]
        curlE: Curl of superposed E-field [3, Nx, Ny, Nz]
    """
    # Initialize with zeros
    E_total = np.zeros((3, Nx, Ny, Nz), dtype=np.float32)
    
    # Superpose multiple plane waves
    for _ in range(num_waves):
        E_single = generate_single_plane_wave()
        E_total += E_single
    
    # Compute curl of the superposition
    curlE = compute_curl_finite_difference(E_total)
    
    return E_total.astype(np.float32), curlE.astype(np.float32)

# For backward compatibility, rename original function
generate_plane_wave = generate_plane_wave_superposition

# -------------------------------
# Dataset with Coordinate Channels
# -------------------------------
class EMFieldDataset(Dataset):
    """
    Dataset for E-field to curl(E) mapping.
    
    Key features:
    1. Includes normalized coordinate channels (x/L, y/L, z/L)
    2. Per-component RMS normalization of curl
    3. Statistics saved for inference
    """
    def __init__(self, num_samples, norm_stats=None, save_stats=True):
        self.data = []
        self.norm_stats = norm_stats
        self.save_stats = save_stats
        
        print(f"Generating {num_samples} samples...")
        
        # Create normalized coordinate grid (will be concatenated with E-field)
        x_norm = np.linspace(0, 1, Nx)
        y_norm = np.linspace(0, 1, Ny)
        z_norm = np.linspace(0, 1, Nz)
        X_norm, Y_norm, Z_norm = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')
        
        # Stack normalized coordinates
        self.coords = np.stack([X_norm, Y_norm, Z_norm], axis=0).astype(np.float32)
        
        # Collect all data
        E_fields = []
        curl_fields = []
        
        for _ in range(num_samples):
            E, curlE = generate_plane_wave()
            E_fields.append(E)
            curl_fields.append(curlE)
        
        # Compute dataset-level normalization if not provided
        if self.norm_stats is None:
            curl_array = np.stack(curl_fields, axis=0)  # [N, 3, Nx, Ny, Nz]
            
            # ---------- KEY FIX: Per-component RMS normalization ----------
            self.norm_stats = {}
            for c in range(3):
                # RMS across spatial dimensions and samples
                rms = np.sqrt(np.mean(curl_array[:, c]**2))
                self.norm_stats[f'curl_{c}_rms'] = float(rms)
            
            print("Computed normalization statistics:")
            print(f"  Curl_x RMS: {self.norm_stats['curl_0_rms']:.6e}")
            print(f"  Curl_y RMS: {self.norm_stats['curl_1_rms']:.6e}")
            print(f"  Curl_z RMS: {self.norm_stats['curl_2_rms']:.6e}")
            
            # Save statistics for inference
            if self.save_stats:
                sigma = np.array([self.norm_stats[f'curl_{c}_rms'] for c in range(3)])
                np.save('outputs/curl_sigma.npy', sigma)
                print(f"Saved normalization constants to outputs/curl_sigma.npy")
        
        # Normalize and store
        for E, curlE in zip(E_fields, curl_fields):
            # ---------- KEY FIX: Add coordinate channels to input ----------
            # Input shape becomes [6, Nx, Ny, Nz]: 3 coords + 3 E-field components
            input_with_coords = np.concatenate([self.coords, E], axis=0)
            
            # Normalize curl using RMS statistics
            curlE_norm = curlE.copy()
            for c in range(3):
                curlE_norm[c] = curlE[c] / (self.norm_stats[f'curl_{c}_rms'] + 1e-10)
            
            self.data.append((input_with_coords, curlE_norm))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        E_input, curlE = self.data[idx]
        return torch.from_numpy(E_input), torch.from_numpy(curlE)
    
    def get_norm_stats(self):
        return self.norm_stats

# -------------------------------
# Model Architecture (Updated for 6 input channels)
# -------------------------------
class ResidualBlock(nn.Module):
    """3D residual block with GELU activation"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.gelu2 = nn.GELU()
    
    def forward(self, x):
        residual = x
        out = self.gelu1(self.conv1(x))
        out = self.gelu2(self.conv2(out))
        return out + residual

class DCO_UNet3D(nn.Module):
    """
    3D U-Net for curl operator learning.
    
    Key updates:
    1. Input channels: 6 (3 coords + 3 E-field components)
    2. Residual blocks for better gradient flow
    3. Output channels: 3 (curl vector components)
    """
    def __init__(self):
        super().__init__()
        
        # Feature channels at each level
        features = [32, 64, 128, 256]
        
        # ---------- Encoder ----------
        # ---------- KEY FIX: 6 input channels instead of 3 ----------
        self.enc1_conv = nn.Sequential(
            nn.Conv3d(6, features[0], kernel_size=3, padding=1),  # Changed from 3 to 6
            nn.GELU(),
            ResidualBlock(features[0])
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2_conv = nn.Sequential(
            nn.Conv3d(features[0], features[1], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[1])
        )
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3_conv = nn.Sequential(
            nn.Conv3d(features[1], features[2], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[2])
        )
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4_conv = nn.Sequential(
            nn.Conv3d(features[2], features[3], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[3])
        )
        self.pool4 = nn.MaxPool3d(2)
        
        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features[3], features[3]*2, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[3]*2)
        )
        
        # ---------- Decoder ----------
        self.up4 = nn.ConvTranspose3d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4_conv = nn.Sequential(
            nn.Conv3d(features[3]*2, features[3], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[3])
        )
        
        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv3d(features[2]*2, features[2], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[2])
        )
        
        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv3d(features[1]*2, features[1], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[1])
        )
        
        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv3d(features[0]*2, features[0], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[0])
        )
        
        # Final convolution to 3 output channels (curl components)
        self.final_conv = nn.Conv3d(features[0], 3, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        e1 = self.enc1_conv(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2_conv(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3_conv(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4_conv(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4_conv(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_conv(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_conv(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_conv(d1)
        
        return self.final_conv(d1)

# -------------------------------
# Combined Loss Function
# -------------------------------
class CombinedLoss(nn.Module):
    """
    Combined loss for vector field regression.
    
    Components:
    1. MSE loss: L2 distance
    2. Sign-sensitive loss: 1 - cosine similarity
    
    The sign-sensitive loss helps prevent vector inversion.
    """
    def __init__(self, lambda_sign=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_sign = lambda_sign
    
    def forward(self, pred, target):
        # MSE loss (magnitude accuracy)
        loss_mse = self.mse(pred, target)
        
        # Sign-sensitive loss (direction accuracy)
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        dot_product = (pred_flat * target_flat).sum(dim=1)
        pred_norm = torch.sqrt((pred_flat ** 2).sum(dim=1) + 1e-10)
        target_norm = torch.sqrt((target_flat ** 2).sum(dim=1) + 1e-10)
        
        cosine_sim = dot_product / (pred_norm * target_norm)
        loss_sign = 1.0 - cosine_sim.mean()
        
        # Combined loss
        loss_total = loss_mse + self.lambda_sign * loss_sign
        
        return loss_total, loss_mse, loss_sign

# -------------------------------
# Training Function
# -------------------------------
def train_model():
    """Main training loop for DCO-UNet3D"""
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Generate datasets
    print("\n" + "="*50)
    print("Creating Datasets")
    print("="*50)
    
    full_dataset = EMFieldDataset(num_samples=1000, save_stats=True)
    norm_stats = full_dataset.get_norm_stats()
    
    # Split: 80% train, 10% val, 10% test
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Setup model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    
    model = DCO_UNet3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CombinedLoss(lambda_sign=0.1)
    
    # Training history
    history = {
        'train_loss': [],
        'train_mse': [],
        'train_sign': [],
        'val_loss': [],
        'val_mse': [],
        'val_sign': []
    }
    
    # Hyperparameters
    config = {
        'epochs': 1000,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'lambda_sign': 0.1,
        'grid_size': Nx,
        'domain_length': L,
        'normalization': norm_stats,
        'input_channels': 6,  # 3 coords + 3 E-field components
        'output_channels': 3   # curl components
    }
    
    epochs = config['epochs']
    
    print("\n" + "="*50)
    print("Training Started")
    print("="*50)
    
    for epoch in range(epochs):
        # ----- Training phase -----
        model.train()
        train_loss_epoch = 0.0
        train_mse_epoch = 0.0
        train_sign_epoch = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss, loss_mse, loss_sign = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_mse_epoch += loss_mse.item()
            train_sign_epoch += loss_sign.item()
        
        # Average losses
        train_loss_epoch /= len(train_loader)
        train_mse_epoch /= len(train_loader)
        train_sign_epoch /= len(train_loader)
        
        # ----- Validation phase -----
        model.eval()
        val_loss_epoch = 0.0
        val_mse_epoch = 0.0
        val_sign_epoch = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss, loss_mse, loss_sign = criterion(outputs, targets)
                
                val_loss_epoch += loss.item()
                val_mse_epoch += loss_mse.item()
                val_sign_epoch += loss_sign.item()
        
        val_loss_epoch /= len(val_loader)
        val_mse_epoch /= len(val_loader)
        val_sign_epoch /= len(val_loader)
        
        # Store history
        history['train_loss'].append(train_loss_epoch)
        history['train_mse'].append(train_mse_epoch)
        history['train_sign'].append(train_sign_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['val_mse'].append(val_mse_epoch)
        history['val_sign'].append(val_sign_epoch)
        
        # Progress update
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss_epoch:.6f} | "
                  f"Val Loss: {val_loss_epoch:.6f}")
    
    print("\n" + "="*50)
    print("Training Complete")
    print("="*50)
    
    # Save model and configuration
    torch.save(model.state_dict(), 'outputs/dco_unet3d.pth')
    with open('outputs/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Model saved to: outputs/dco_unet3d.pth")
    print("Config saved to: outputs/config.json")
    
    return model, test_loader, device, history, norm_stats

# -------------------------------
# Standalone Inference Function
# -------------------------------
def run_inference(E_input_np, model_path='outputs/dco_unet3d.pth'):
    """
    Standalone inference function for curl prediction.
    
    Args:
        E_input_np: numpy array of shape [3, Nx, Ny, Nz] (E-field components)
        model_path: path to trained model
    
    Returns:
        curl_pred: numpy array of shape [3, Nx, Ny, Nz] (predicted curl)
    """
    # Load normalization constants
    sigma = np.load('outputs/curl_sigma.npy')
    sigma_tensor = torch.from_numpy(sigma).float()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCO_UNet3D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create normalized coordinate grid
    x_norm = np.linspace(0, 1, Nx)
    y_norm = np.linspace(0, 1, Ny)
    z_norm = np.linspace(0, 1, Nz)
    X_norm, Y_norm, Z_norm = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')
    coords = np.stack([X_norm, Y_norm, Z_norm], axis=0).astype(np.float32)
    
    # Concatenate coordinates with E-field
    input_with_coords = np.concatenate([coords, E_input_np], axis=0)
    input_tensor = torch.from_numpy(input_with_coords).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        curl_pred_norm = model(input_tensor)
        
        # Un-normalize using saved constants
        sigma_tensor = sigma_tensor.to(device)
        curl_pred = curl_pred_norm * sigma_tensor[None, :, None, None, None]
    
    return curl_pred[0].cpu().numpy()

# -------------------------------
# Evaluation and Visualization
# -------------------------------
def evaluate_and_visualize(model, test_loader, device, history, norm_stats):
    """Comprehensive evaluation and visualization"""
    model.eval()
    
    # 1. Plot training curves
    print("\n" + "="*50)
    print("Generating Training Curves")
    print("="*50)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss vs Epoch')
    axes[0].legend()
    axes[0].grid(True)
    
    # MSE loss
    axes[1].plot(epochs, history['train_mse'], label='Train MSE')
    axes[1].plot(epochs, history['val_mse'], label='Val MSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('MSE Loss vs Epoch')
    axes[1].legend()
    axes[1].grid(True)
    
    # Sign loss
    axes[2].plot(epochs, history['train_sign'], label='Train Sign')
    axes[2].plot(epochs, history['val_sign'], label='Val Sign')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Sign Loss')
    axes[2].set_title('Sign Loss vs Epoch')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    print("Saved: outputs/training_curves.png")
    plt.close()
    
    # 2. Component-wise error analysis
    print("\n" + "="*50)
    print("Component-wise Error Analysis")
    print("="*50)
    
    with torch.no_grad():
        all_targets = []
        all_preds = []
        
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
        
        targets_array = np.concatenate(all_targets, axis=0)
        preds_array = np.concatenate(all_preds, axis=0)
        
        # Un-normalize for physical error computation
        for c in range(3):
            targets_array[:, c] *= norm_stats[f'curl_{c}_rms']
            preds_array[:, c] *= norm_stats[f'curl_{c}_rms']
        
        # Component-wise errors
        component_names = ['x', 'y', 'z']
        print("\nComponent-wise errors:")
        for c, name in enumerate(component_names):
            mse_c = np.mean((preds_array[:, c] - targets_array[:, c])**2)
            rel_l2 = np.linalg.norm(preds_array[:, c] - targets_array[:, c]) / \
                     (np.linalg.norm(targets_array[:, c]) + 1e-10)
            print(f"  Component {name}: MSE = {mse_c:.6e}, Relative L2 = {rel_l2:.6f}")
        
        # Vector direction check (sign verification)
        print("\nSign verification (should be positive):")
        for i in range(min(5, len(targets_array))):
            dot = np.sum(preds_array[i] * targets_array[i])
            print(f"  Sample {i}: Dot product = {dot:.6e}")
    
    # 3. Field slice visualization
    print("\n" + "="*50)
    print("Generating Field Slice Visualizations")
    print("="*50)
    
    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # Take first sample and un-normalize
    target_curl = targets[0].cpu().numpy()
    pred_curl = outputs[0].cpu().numpy()
    
    for c in range(3):
        target_curl[c] *= norm_stats[f'curl_{c}_rms']
        pred_curl[c] *= norm_stats[f'curl_{c}_rms']
    
    mid = Nx // 2
    
    # XY slice visualization
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    vmax = max(np.abs(target_curl).max(), np.abs(pred_curl).max())
    
    for c, name in enumerate(component_names):
        # Ground truth
        im0 = axes[0, c].imshow(target_curl[c, :, :, mid], cmap='RdBu_r', 
                                vmin=-vmax, vmax=vmax)
        axes[0, c].set_title(f'GT Curl {name} (XY slice, z={mid})')
        plt.colorbar(im0, ax=axes[0, c])
        
        # Prediction
        im1 = axes[1, c].imshow(pred_curl[c, :, :, mid], cmap='RdBu_r', 
                                vmin=-vmax, vmax=vmax)
        axes[1, c].set_title(f'Pred Curl {name}')
        plt.colorbar(im1, ax=axes[1, c])
        
        # Absolute error
        error = np.abs(target_curl[c, :, :, mid] - pred_curl[c, :, :, mid])
        im2 = axes[2, c].imshow(error, cmap='inferno')
        axes[2, c].set_title(f'Abs Error {name}')
        plt.colorbar(im2, ax=axes[2, c])
    
    plt.tight_layout()
    plt.savefig('outputs/field_slices_xy.png', dpi=150)
    print("Saved: outputs/field_slices_xy.png")
    plt.close()
    
    # 4. Distribution diagnostics
    print("\n" + "="*50)
    print("Generating Distribution Diagnostics")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram of ground truth
    axes[0, 0].hist(targets_array.flatten(), bins=100, alpha=0.7, label='Ground Truth')
    axes[0, 0].set_xlabel('Curl Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution: Ground Truth Curl')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Histogram of prediction
    axes[0, 1].hist(preds_array.flatten(), bins=100, alpha=0.7, 
                    label='Prediction', color='orange')
    axes[0, 1].set_xlabel('Curl Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution: Predicted Curl')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Scatter plot
    sample_idx = np.random.choice(len(targets_array.flatten()), 10000, replace=False)
    axes[1, 0].scatter(targets_array.flatten()[sample_idx], 
                       preds_array.flatten()[sample_idx], 
                       alpha=0.3, s=1)
    axes[1, 0].plot([targets_array.min(), targets_array.max()], 
                    [targets_array.min(), targets_array.max()], 
                    'r--', label='Perfect prediction')
    axes[1, 0].set_xlabel('Ground Truth')
    axes[1, 0].set_ylabel('Prediction')
    axes[1, 0].set_title('Scatter: Prediction vs Ground Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Overlay histograms
    axes[1, 1].hist(targets_array.flatten(), bins=100, alpha=0.5, label='GT')
    axes[1, 1].hist(preds_array.flatten(), bins=100, alpha=0.5, label='Pred')
    axes[1, 1].set_xlabel('Curl Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/distribution_diagnostics.png', dpi=150)
    print("Saved: outputs/distribution_diagnostics.png")
    plt.close()
    
    print("\n" + "="*50)
    print("Evaluation Complete")
    print("="*50)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DCO-UNet 3D: Differentiable Curl Operator Learning")
    print("="*60)
    
    # Train the model
    model, test_loader, device, history, norm_stats = train_model()
    
    # Evaluate and visualize
    evaluate_and_visualize(model, test_loader, device, history, norm_stats)
    
    # Demonstrate standalone inference
    print("\n" + "="*50)
    print("Testing Standalone Inference")
    print("="*50)
    
    # Create a test E-field
    test_E, test_curl_true = generate_plane_wave()
    
    # Run inference
    curl_pred = run_inference(test_E)
    
    # Compare with true curl
    mse_total = np.mean((curl_pred - test_curl_true)**2)
    print(f"\nInference test:")
    print(f"  MSE between predicted and true curl: {mse_total:.6e}")
    
    # Sign check (should be positive)
    dot_product = np.sum(curl_pred * test_curl_true)
    print(f"  Dot product (sign check): {dot_product:.6e}")
    print(f"  {'✓ No sign inversion' if dot_product > 0 else '✗ Possible sign inversion'}")
    
    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print("="*60)