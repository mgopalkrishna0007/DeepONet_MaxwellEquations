"""
Deep Curl Operator (DCO) - Complete Implementation
A 3D U-Net based neural operator for learning the curl operator from electromagnetic fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path

# ============================================================================
# 1. PHYSICAL DATASET GENERATOR
# ============================================================================

class EMFieldDatasetGenerator:
    """Generates 3D electromagnetic field datasets with analytical curl"""
    
    def __init__(self, grid_size=32, domain_size=19.2e-3):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.c0 = 3e8  # Speed of light
        
    def generate_sample(self, randomize_grid=True):
        """Generate a single training sample"""
        
        # Randomize grid spacing if requested
        if randomize_grid:
            dx = np.random.uniform(0.3e-3, 0.8e-3)
            dy = np.random.uniform(0.3e-3, 0.8e-3)
            dz = np.random.uniform(0.3e-3, 0.8e-3)
        else:
            dx = dy = dz = 0.6e-3
            
        # Create coordinate grid
        N = self.grid_size
        x = np.linspace(0, dx * (N-1), N)
        y = np.linspace(0, dy * (N-1), N)
        z = np.linspace(0, dz * (N-1), N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X, Y, Z], axis=-1)  # [N, N, N, 3]
        
        # Random physical parameters
        theta = np.random.uniform(0, np.pi)        # θ ∈ [0, π]
        phi = np.random.uniform(-np.pi, np.pi)     # ϕ ∈ [-π, π]
        f = np.random.uniform(1e6, 50e9)           # 1 MHz to 50 GHz
        k0 = 2 * np.pi * f / self.c0
        
        # Wave vector
        kx = k0 * np.sin(theta) * np.cos(phi)
        ky = k0 * np.sin(theta) * np.sin(phi)
        kz = k0 * np.cos(theta)
        k = np.array([kx, ky, kz])
        
        # Random transverse polarization (enforcing k·E=0)
        Ex0 = np.random.uniform(0, 5)
        Ey0 = np.random.uniform(0, 5)
        
        if abs(kz) > 1e-6:
            Ez0 = -(kx * Ex0 + ky * Ey0) / kz
        else:
            Ez0 = np.random.uniform(0, 5)
            # Re-adjust to maintain k·E=0
            if abs(kx) > 1e-6:
                Ex0 = -(ky * Ey0 + kz * Ez0) / kx
                
        E0 = np.array([Ex0, Ey0, Ez0])
        
        # Electric field (plane wave)
        phase = kx * X + ky * Y + kz * Z
        Ex = E0[0] * np.cos(phase)
        Ey = E0[1] * np.cos(phase)
        Ez = E0[2] * np.cos(phase)
        E = np.stack([Ex, Ey, Ez], axis=0)  # [3, N, N, N]
        
        # Analytical curl: ∇×E = k × E
        curl_x = ky * Ez - kz * Ey
        curl_y = kz * Ex - kx * Ez
        curl_z = kx * Ey - ky * Ex
        curlE = np.stack([curl_x, curl_y, curl_z], axis=0)  # [3, N, N, N]
        
        return E, curlE, coords, {'k': k, 'freq': f, 'E0': E0}
    
    def generate_dataset(self, num_samples, train_ratio=0.8):
        """Generate full dataset"""
        E_data, curl_data, coord_data = [], [], []
        
        print(f"Generating {num_samples} samples...")
        for i in tqdm(range(num_samples)):
            E, curl, coords, _ = self.generate_sample(randomize_grid=True)
            E_data.append(E)
            curl_data.append(curl)
            coord_data.append(coords)
            
        # Convert to numpy arrays
        E_data = np.array(E_data, dtype=np.float32)
        curl_data = np.array(curl_data, dtype=np.float32)
        coord_data = np.array(coord_data, dtype=np.float32)
        
        # Split into train/test
        split_idx = int(num_samples * train_ratio)
        
        train_data = {
            'E': E_data[:split_idx],
            'curl': curl_data[:split_idx],
            'coords': coord_data[:split_idx]
        }
        
        test_data = {
            'E': E_data[split_idx:],
            'curl': curl_data[split_idx:],
            'coords': coord_data[split_idx:]
        }
        
        return train_data, test_data
    
    def generate_test_set(self, num_samples=20):
        """Generate test set with fixed parameters"""
        test_samples = []
        
        # Fixed parameters
        dx = dy = dz = 0.6e-3
        N = self.grid_size
        x = np.linspace(0, dx * (N-1), N)
        y = np.linspace(0, dy * (N-1), N)
        z = np.linspace(0, dz * (N-1), N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        fixed_coords = np.stack([X, Y, Z], axis=-1)
        
        # Fixed angles
        theta = np.deg2rad(45)
        phi = np.deg2rad(60)
        
        # Uniform wavenumbers
        k_values = np.linspace(0.021, 838.34, num_samples)
        
        print(f"Generating {num_samples} test samples with fixed parameters...")
        for k0 in tqdm(k_values):
            # Wave vector
            kx = k0 * np.sin(theta) * np.cos(phi)
            ky = k0 * np.sin(theta) * np.sin(phi)
            kz = k0 * np.cos(theta)
            k = np.array([kx, ky, kz])
            
            # Fixed polarization (k·E=0)
            Ex0 = 1.0
            Ey0 = 1.0
            Ez0 = -(kx * Ex0 + ky * Ey0) / kz if abs(kz) > 1e-6 else 0.0
            E0 = np.array([Ex0, Ey0, Ez0])
            
            # Electric field
            phase = kx * X + ky * Y + kz * Z
            Ex = E0[0] * np.cos(phase)
            Ey = E0[1] * np.cos(phase)
            Ez = E0[2] * np.cos(phase)
            E = np.stack([Ex, Ey, Ez], axis=0)
            
            # Analytical curl
            curl_x = ky * Ez - kz * Ey
            curl_y = kz * Ex - kx * Ez
            curl_z = kx * Ey - ky * Ex
            curlE = np.stack([curl_x, curl_y, curl_z], axis=0)
            
            test_samples.append({
                'E': E.astype(np.float32),
                'curl': curlE.astype(np.float32),
                'coords': fixed_coords.astype(np.float32),
                'k': k0,
                'freq': k0 * self.c0 / (2 * np.pi)
            })
            
        return test_samples

# ============================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# ============================================================================

class ResidualBlock3D(nn.Module):
    """3D residual block with two convolutions"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act = nn.GELU()
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)

class EncoderBlock3D(nn.Module):
    """Encoder block with residual connections and pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResidualBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        features = self.res_block(x)
        pooled = self.pool(features)
        return features, pooled

class DecoderBlock3D(nn.Module):
    """Decoder block with upsampling and skip connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.res_block = ResidualBlock3D(in_channels//2 + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        # Ensure spatial dimensions match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x)

class TrunkNetwork(nn.Module):
    """DeepONet-style trunk network for coordinate encoding"""
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, coords):
        # coords: [B, 3, X, Y, Z]
        B, C, X, Y, Z = coords.shape
        # Flatten spatial dimensions
        coords_flat = coords.permute(0, 2, 3, 4, 1).reshape(B * X * Y * Z, C)
        # Process through network
        features = self.network(coords_flat)
        # Reshape back to spatial dimensions
        features = features.reshape(B, X, Y, Z, -1).permute(0, 4, 1, 2, 3)
        return features

class DeepCurlOperator(nn.Module):
    """Deep Curl Operator: 3D U-Net with DeepONet-style trunk network"""
    
    def __init__(self, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        
        # Input: [B, 6, X, Y, Z] where channels are [x, y, z, Ex, Ey, Ez]
        input_channels = 6
        
        # Encoder path
        self.enc1 = EncoderBlock3D(input_channels, 64)
        self.enc2 = EncoderBlock3D(64, 128)
        self.enc3 = EncoderBlock3D(128, 256)
        self.enc4 = EncoderBlock3D(256, 512)
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(512, 1024)
        
        # Decoder path
        self.dec4 = DecoderBlock3D(1024, 512, 512)
        self.dec3 = DecoderBlock3D(512, 256, 256)
        self.dec2 = DecoderBlock3D(256, 128, 128)
        self.dec1 = DecoderBlock3D(128, 64, 64)
        
        # Trunk network (DeepONet-style)
        self.trunk = TrunkNetwork(input_dim=3, hidden_dims=[32, 64, 128])
        
        # Final projection
        self.final = nn.Sequential(
            nn.Conv3d(64 + 128, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(32, 3, kernel_size=1)  # Output: curl_x, curl_y, curl_z
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Args:
            x: [B, 6, X, Y, Z] where channels are [x, y, z, Ex, Ey, Ez]
        Returns:
            curl: [B, 3, X, Y, Z] curl components
        """
        # Split input into coordinates and field
        coords = x[:, :3, :, :, :]  # [B, 3, X, Y, Z]
        
        # Encode coordinates with trunk network
        trunk_features = self.trunk(coords)  # [B, 128, X, Y, Z]
        
        # U-Net encoder path
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # U-Net decoder path with skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        
        # Combine with trunk features
        combined = torch.cat([d1, trunk_features], dim=1)
        
        # Final projection
        curl = self.final(combined)
        
        return curl

# ============================================================================
# 3. LOSS FUNCTIONS
# ============================================================================

class CurlLoss(nn.Module):
    """Custom loss function for curl operator learning"""
    def __init__(self, normalization='max'):
        super().__init__()
        self.normalization = normalization
        
    def forward(self, pred, target):
        if self.normalization == 'max':
            # Component-wise max normalization
            max_vals = target.abs().amax(dim=(2, 3, 4), keepdim=True) + 1e-8
            pred_norm = pred / max_vals
            target_norm = target / max_vals
            return F.mse_loss(pred_norm, target_norm)
        else:
            # No normalization
            return F.mse_loss(pred, target)

class MaxwellConstraintLoss(nn.Module):
    """Physics-informed loss enforcing Maxwell's equations"""
    def __init__(self, dx=0.6e-3):
        super().__init__()
        self.dx = dx
        
    def finite_difference_curl(self, E):
        """Compute curl using central finite differences"""
        # E: [B, 3, X, Y, Z]
        B, C, X, Y, Z = E.shape
        
        # Pad for boundary conditions
        E_padded = F.pad(E, (1, 1, 1, 1, 1, 1), mode='replicate')
        
        curl = torch.zeros_like(E)
        
        # curl_x = ∂Ez/∂y - ∂Ey/∂z
        curl[:, 0] = (E_padded[:, 2, :, 2:-1, 1:-2] - E_padded[:, 2, :, 0:-3, 1:-2]) / (2*self.dx) \
                   - (E_padded[:, 1, :, 1:-2, 2:-1] - E_padded[:, 1, :, 1:-2, 0:-3]) / (2*self.dx)
        
        # curl_y = ∂Ex/∂z - ∂Ez/∂x
        curl[:, 1] = (E_padded[:, 0, :, 1:-2, 2:-1] - E_padded[:, 0, :, 1:-2, 0:-3]) / (2*self.dx) \
                   - (E_padded[:, 2, :, 2:-1, 1:-2] - E_padded[:, 2, :, 0:-3, 1:-2]) / (2*self.dx)
        
        # curl_z = ∂Ey/∂x - ∂Ex/∂y
        curl[:, 2] = (E_padded[:, 1, :, 2:-1, 1:-2] - E_padded[:, 1, :, 0:-3, 1:-2]) / (2*self.dx) \
                   - (E_padded[:, 0, :, 1:-2, 2:-1] - E_padded[:, 0, :, 1:-2, 0:-3]) / (2*self.dx)
        
        return curl
        
    def forward(self, pred_curl, E_field):
        """Compute physics constraint loss"""
        fd_curl = self.finite_difference_curl(E_field)
        return F.mse_loss(pred_curl, fd_curl)

# ============================================================================
# 4. TRAINER CLASS
# ============================================================================

class DCO_Trainer:
    """Training wrapper for Deep Curl Operator"""
    
    def __init__(self, model, device='cuda', lr=1e-4):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.data_loss_fn = CurlLoss(normalization='max')
        self.physics_loss_fn = MaxwellConstraintLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mre': [],
            'val_mre': []
        }
        
    def prepare_input(self, E_field, coords):
        """Prepare input tensor [B, 6, X, Y, Z]"""
        B = E_field.shape[0]
        # Ensure coordinates are in correct format
        if coords.ndim == 4:  # [X, Y, Z, 3]
            coords = torch.FloatTensor(coords).unsqueeze(0).repeat(B, 1, 1, 1, 1)
            coords = coords.permute(0, 4, 1, 2, 3)  # [B, 3, X, Y, Z]
        elif coords.ndim == 5:  # [B, 3, X, Y, Z]
            coords = torch.FloatTensor(coords)
            
        input_tensor = torch.cat([coords, E_field], dim=1)
        return input_tensor.to(self.device)
    
    def compute_metrics(self, pred, target):
        """Compute various metrics"""
        metrics = {}
        
        # MSE
        metrics['mse'] = F.mse_loss(pred, target).item()
        
        # MAE
        metrics['mae'] = F.l1_loss(pred, target).item()
        
        # Mean Relative Error (MRE)
        abs_error = torch.abs(pred - target)
        rel_error = abs_error / (torch.abs(target) + 1e-8)
        metrics['mre'] = rel_error.mean().item()
        
        # Max Relative Error
        metrics['max_re'] = rel_error.max().item()
        
        return metrics
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mre = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            E_batch = batch['E'].to(self.device)
            curl_batch = batch['curl'].to(self.device)
            coords_batch = batch['coords']
            
            # Prepare input
            input_tensor = self.prepare_input(E_batch, coords_batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_curl = self.model(input_tensor)
            
            # Compute losses
            data_loss = self.data_loss_fn(pred_curl, curl_batch)
            physics_loss = self.physics_loss_fn(pred_curl, E_batch)
            loss = data_loss + 0.1 * physics_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Compute metrics
            metrics = self.compute_metrics(pred_curl, curl_batch)
            
            total_loss += loss.item()
            total_mre += metrics['mre']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mre': f'{metrics["mre"]:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
        avg_loss = total_loss / num_batches
        avg_mre = total_mre / num_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_mre'].append(avg_mre)
        
        return avg_loss, avg_mre
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mre = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                E_batch = batch['E'].to(self.device)
                curl_batch = batch['curl'].to(self.device)
                coords_batch = batch['coords']
                
                # Prepare input
                input_tensor = self.prepare_input(E_batch, coords_batch)
                
                # Forward pass
                pred_curl = self.model(input_tensor)
                
                # Compute loss and metrics
                loss = self.data_loss_fn(pred_curl, curl_batch)
                metrics = self.compute_metrics(pred_curl, curl_batch)
                
                total_loss += loss.item()
                total_mre += metrics['mre']
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mre = total_mre / num_batches
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_mre'].append(avg_mre)
        
        return avg_loss, avg_mre
    
    def train(self, train_loader, val_loader, num_epochs=1000):
        """Full training loop"""
        print("Starting training...")
        best_val_mre = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_mre = self.train_epoch(train_loader, epoch)
            
            # Validate
            if epoch % 10 == 0:
                val_loss, val_mre = self.validate(val_loader)
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f}, Train MRE: {train_mre:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val MRE: {val_mre:.4f}")
                
                # Save best model
                if val_mre < best_val_mre:
                    best_val_mre = val_mre
                    self.save_model(f'best_model.pth')
                    print(f"  Saved best model with MRE: {val_mre:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
        print(f"\nTraining complete! Best validation MRE: {best_val_mre:.4f}")
        
        # Plot training history
        self.plot_training_history()
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MRE plot
        axes[1].plot(self.history['train_mre'], label='Train MRE')
        axes[1].plot(self.history['val_mre'], label='Val MRE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Relative Error')
        axes[1].set_title('Training History - MRE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# 5. DATA LOADER
# ============================================================================

class EMDataLoader:
    """Custom data loader for EM field data"""
    
    def __init__(self, data_dict, batch_size=32, shuffle=True):
        self.E_data = torch.FloatTensor(data_dict['E'])
        self.curl_data = torch.FloatTensor(data_dict['curl'])
        self.coords_data = torch.FloatTensor(data_dict['coords'])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.E_data)
        
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
        else:
            indices = torch.arange(self.num_samples)
            
        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            yield {
                'E': self.E_data[batch_idx],
                'curl': self.curl_data[batch_idx],
                'coords': self.coords_data[batch_idx]
            }
            
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

# ============================================================================
# 6. VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_results(model, test_sample, device='cuda'):
    """Visualize predictions for a test sample"""
    model.eval()
    
    # Prepare input
    E = torch.FloatTensor(test_sample['E']).unsqueeze(0).to(device)
    coords = torch.FloatTensor(test_sample['coords']).unsqueeze(0).to(device)
    input_tensor = torch.cat([coords.permute(0, 4, 1, 2, 3), E], dim=1)
    
    # Get prediction
    with torch.no_grad():
        pred_curl = model(input_tensor).cpu().numpy()[0]
    
    # Ground truth
    gt_curl = test_sample['curl']
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    titles = ['Curl_x', 'Curl_y', 'Curl_z']
    
    # Choose middle slice for visualization
    slice_idx = 16
    
    for i in range(3):
        # Ground truth
        im1 = axes[i, 0].imshow(gt_curl[i, slice_idx, :, :], cmap='RdBu_r')
        axes[i, 0].set_title(f'GT {titles[i]}')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Prediction
        im2 = axes[i, 1].imshow(pred_curl[i, slice_idx, :, :], cmap='RdBu_r')
        axes[i, 1].set_title(f'Pred {titles[i]}')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Absolute error
        abs_error = np.abs(pred_curl[i] - gt_curl[i])
        im3 = axes[i, 2].imshow(abs_error[slice_idx, :, :], cmap='hot')
        axes[i, 2].set_title(f'Abs Error {titles[i]}')
        plt.colorbar(im3, ax=axes[i, 2])
        
        # Relative error
        rel_error = abs_error / (np.abs(gt_curl[i]) + 1e-8)
        im4 = axes[i, 3].imshow(rel_error[slice_idx, :, :], cmap='hot', vmax=0.1)
        axes[i, 3].set_title(f'Rel Error {titles[i]}')
        plt.colorbar(im4, ax=axes[i, 3])
    
    plt.suptitle(f'Test Sample: k={test_sample["k"]:.2f} rad/m, f={test_sample["freq"]/1e9:.2f} GHz')
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    mse = np.mean((pred_curl - gt_curl) ** 2)
    mae = np.mean(np.abs(pred_curl - gt_curl))
    mre = np.mean(np.abs(pred_curl - gt_curl) / (np.abs(gt_curl) + 1e-8))
    
    print(f"Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MRE: {mre:.6f}")

def plot_mre_vs_wavenumber(model, test_set, device='cuda'):
    """Plot MRE vs wavenumber for test set"""
    model.eval()
    
    wavenumbers = []
    mre_values = []
    
    for sample in test_set:
        # Prepare input
        E = torch.FloatTensor(sample['E']).unsqueeze(0).to(device)
        coords = torch.FloatTensor(sample['coords']).unsqueeze(0).to(device)
        input_tensor = torch.cat([coords.permute(0, 4, 1, 2, 3), E], dim=1)
        
        # Get prediction
        with torch.no_grad():
            pred_curl = model(input_tensor).cpu().numpy()[0]
        
        # Compute MRE
        gt_curl = sample['curl']
        rel_error = np.abs(pred_curl - gt_curl) / (np.abs(gt_curl) + 1e-8)
        mre = np.mean(rel_error)
        
        wavenumbers.append(sample['k'])
        mre_values.append(mre)
    
    # Sort by wavenumber
    sort_idx = np.argsort(wavenumbers)
    wavenumbers = np.array(wavenumbers)[sort_idx]
    mre_values = np.array(mre_values)[sort_idx]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(wavenumbers, mre_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Wavenumber (rad/m)')
    plt.ylabel('Mean Relative Error')
    plt.title('MRE vs Wavenumber (θ=45°, φ=60°)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mre_vs_wavenumber.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Average MRE across test set: {np.mean(mre_values):.4f}")
    print(f"Max MRE: {np.max(mre_values):.4f}")
    print(f"Min MRE: {np.min(mre_values):.4f}")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # ========================================================================
    # Step 1: Generate dataset
    # ========================================================================
    print("\n" + "="*60)
    print("Step 1: Generating dataset")
    print("="*60)
    
    generator = EMFieldDatasetGenerator(grid_size=32, domain_size=19.2e-3)
    train_data, test_data = generator.generate_dataset(num_samples=1000, train_ratio=0.8)
    
    print(f"Training samples: {len(train_data['E'])}")
    print(f"Test samples: {len(test_data['E'])}")
    
    # ========================================================================
    # Step 2: Create data loaders
    # ========================================================================
    print("\n" + "="*60)
    print("Step 2: Creating data loaders")
    print("="*60)
    
    train_loader = EMDataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = EMDataLoader(test_data, batch_size=32, shuffle=False)
    
    # ========================================================================
    # Step 3: Initialize model
    # ========================================================================
    print("\n" + "="*60)
    print("Step 3: Initializing Deep Curl Operator")
    print("="*60)
    
    model = DeepCurlOperator(grid_size=32)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ========================================================================
    # Step 4: Initialize trainer
    # ========================================================================
    print("\n" + "="*60)
    print("Step 4: Initializing trainer")
    print("="*60)
    
    trainer = DCO_Trainer(model, device=device, lr=1e-4)
    
    # ========================================================================
    # Step 5: Train model
    # ========================================================================
    print("\n" + "="*60)
    print("Step 5: Training model")
    print("="*60)
    
    # Train for a smaller number of epochs for demonstration
    # Increase to 1000 for full training
    trainer.train(train_loader, val_loader, num_epochs=100)
    
    # ========================================================================
    # Step 6: Generate test set and evaluate
    # ========================================================================
    print("\n" + "="*60)
    print("Step 6: Evaluating on test set")
    print("="*60)
    
    # Generate test set with fixed parameters
    test_set = generator.generate_test_set(num_samples=20)
    
    # Load best model
    trainer.load_model('best_model.pth')
    
    # Visualize first test sample
    print("\nVisualizing first test sample...")
    visualize_results(model, test_set[0], device=device)
    
    # Plot MRE vs wavenumber
    print("\nPlotting MRE vs wavenumber...")
    plot_mre_vs_wavenumber(model, test_set, device=device)
    
    # ========================================================================
    # Step 7: Save final model
    # ========================================================================
    print("\n" + "="*60)
    print("Step 7: Saving final model")
    print("="*60)
    
    trainer.save_model('final_model.pth')
    print("Model saved as 'final_model.pth'")
    
    print("\n" + "="*60)
    print("DEEP CURL OPERATOR TRAINING COMPLETE!")
    print("="*60)

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install required packages:")
        print("pip install torch numpy matplotlib tqdm")
        exit(1)
    
    # Run the main function
    main()