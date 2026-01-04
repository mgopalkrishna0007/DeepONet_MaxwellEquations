"""
DCO-UNet 3D: Differentiable Curl Operator via Deep Learning
Physics-informed 3D convolutional neural network for learning curl operator from E-field to curl(E)

Based on paper: "A Differentiable Curl Operator for Deep Learning Electromagnetics"

Key Features:
1. Modified U-net with DeepONet-style trunk network for coordinate processing
2. 3D convolutions with residual connections
3. Superposition of multiple plane waves (n=20) as per theory
4. Analytical curl computation as per Maxwell's equations
5. Proper normalization by local maximum for each component
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# ============================================================================
# PHYSICAL CONSTANTS AND CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for physical and training parameters"""
    # Physical constants
    c0 = 3e8  # Speed of light (m/s)
    pi = np.pi
    
    # Grid parameters (32x32x32 as per paper)
    Nx = Ny = Nz = 32
    N_CELLS = 32  # Number of cells in each dimension
    
    # Cell dimensions range (0.3-0.8 mm as per paper)
    DELTA_MIN = 0.3e-3  # 0.3 mm
    DELTA_MAX = 0.8e-3  # 0.8 mm
    
    # Wave parameters as per theory
    NUM_WAVES = 20  # n=20 superimposed waves
    THETA_RANGE = (0, np.pi)  # 0 to π
    PHI_RANGE = (-np.pi, np.pi)  # -π to +π
    K_MIN = 0  # k minimum
    K_MAX = 1048  # k maximum (results in 0-50 GHz)
    AMPLITUDE_RANGE = (0, 5)  # Ex, Ey amplitudes
    
    # Dataset parameters
    DATASET_SIZE = 1000
    TRAIN_RATIO = 0.8  # 80/20 train-test split
    BATCH_SIZE = 32
    
    # Training parameters
    EPOCHS = 1000  # As per paper
    LEARNING_RATE = 1e-4  # As per paper
    VAL_RATIO = 0.1  # 10% validation
    
    # Model parameters
    INIT_CHANNELS = 32
    NUM_LEVELS = 4
    
    # Output directories
    OUTPUT_DIR = Path("outputs")
    MODEL_DIR = Path("models")
    FIGURES_DIR = Path("figures")
    
    # Inference test case (from paper)
    INFERENCE_DELTA = 0.6e-3  # 0.6 mm
    INFERENCE_THETA = np.deg2rad(45)  # 45 degrees
    INFERENCE_PHI = np.deg2rad(60)  # 60 degrees
    INFERENCE_K_VALUES = 20  # 20 values uniformly selected
    INFERENCE_K_MIN = 0.021
    INFERENCE_K_MAX = 838.34

config = Config()

# ============================================================================
# COORDINATE GRID GENERATION
# ============================================================================

class CoordinateGrid3D:
    """Manages 3D coordinate grid generation with variable cell sizes"""
    
    def __init__(self, delta_x: float, delta_y: float, delta_z: float):
        """
        Initialize grid with specific cell dimensions.
        
        Args:
            delta_x, delta_y, delta_z: Cell dimensions in meters (0.3-0.8 mm range)
        """
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
        
        # Generate coordinates at cell centers
        self.x = np.linspace(delta_x/2, config.N_CELLS * delta_x - delta_x/2, config.Nx)
        self.y = np.linspace(delta_y/2, config.N_CELLS * delta_y - delta_y/2, config.Ny)
        self.z = np.linspace(delta_z/2, config.N_CELLS * delta_z - delta_z/2, config.Nz)
        
        # Create meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Normalized coordinates (0 to 1)
        self.X_norm = self.X / (config.N_CELLS * delta_x)
        self.Y_norm = self.Y / (config.N_CELLS * delta_y)
        self.Z_norm = self.Z / (config.N_CELLS * delta_z)
        
        # Coordinate deltas for trunk input (normalized)
        self.dx_norm = delta_x / (config.N_CELLS * delta_x) * np.ones_like(self.X_norm)
        self.dy_norm = delta_y / (config.N_CELLS * delta_y) * np.ones_like(self.Y_norm)
        self.dz_norm = delta_z / (config.N_CELLS * delta_z) * np.ones_like(self.Z_norm)
    
    def get_physical_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return physical coordinates"""
        return self.X, self.Y, self.Z
    
    def get_coordinate_deltas(self) -> np.ndarray:
        """Return normalized coordinate deltas as [3, Nx, Ny, Nz] array"""
        return np.stack([self.dx_norm, self.dy_norm, self.dz_norm], axis=0).astype(np.float32)
    
    def get_cell_dimensions(self) -> Tuple[float, float, float]:
        """Return cell dimensions"""
        return self.delta_x, self.delta_y, self.delta_z


    def get_trunk_input(self):
        Xn = self.X / self.X.max()
        Yn = self.Y / self.Y.max()
        Zn = self.Z / self.Z.max()
        return np.stack([Xn, Yn, Zn], axis=0).astype(np.float32)

# ============================================================================
# PHYSICS: PLANE WAVE GENERATION AND ANALYTICAL CURL COMPUTATION
# ============================================================================

class PlaneWaveGenerator3D:
    """Generates plane wave superpositions as per paper specifications"""
    
    @staticmethod
    def generate_wave_parameters(num_waves: int = config.NUM_WAVES) -> Dict:
        """
        Generate wave parameters as per paper:
        - Random theta in [0, π]
        - Random phi in [-π, π]
        - For fixed theta, phi: multiple k values between 0 and 1048
        """
        # Single theta and phi for all waves in this sample (as per paper)
        theta = np.random.uniform(*config.THETA_RANGE)
        phi = np.random.uniform(*config.PHI_RANGE)
        
        # Multiple k values (n=20)
        k_values = np.random.uniform(config.K_MIN, config.K_MAX, num_waves)
        
        # Random amplitudes for Ex and Ey
        Ex0_values = np.random.uniform(*config.AMPLITUDE_RANGE, num_waves)
        Ey0_values = np.random.uniform(*config.AMPLITUDE_RANGE, num_waves)
        
        return {
            'theta': theta,
            'phi': phi,
            'k_values': k_values,
            'Ex0': Ex0_values,
            'Ey0': Ey0_values
        }
    
    @staticmethod
    def generate_single_wave(grid: CoordinateGrid3D, theta: float, phi: float, 
                           k: float, Ex0: float, Ey0: float) -> np.ndarray:
        """
        Generate a single plane wave as per equation:
        E_i = E0,i exp[-j k_i (x cosφ sinθ + y sinφ sinθ + z cosθ)]
        
        Returns:
            E: [3, Nx, Ny, Nz] array of E-field components
        """
        # Wave vector components
        kx = k * np.sin(theta) * np.cos(phi)
        ky = k * np.sin(theta) * np.sin(phi)
        kz = k * np.cos(theta)
        
        # Calculate Ez to satisfy ∇·E = 0 (ki·E0,i = 0)
        if np.abs(np.cos(theta)) > 1e-10:
            Ez0 = -(np.cos(phi) * np.sin(theta) * Ex0 + 
                    np.sin(phi) * np.sin(theta) * Ey0) / np.cos(theta)
        else:
            Ez0 = 0.0
        
        # Phase term: k·r = kx*x + ky*y + kz*z
        phase = kx * grid.X + ky * grid.Y + kz * grid.Z
        
        # Complex exponential (using Euler's formula for real part)
        E = np.zeros((3, config.Nx, config.Ny, config.Nz), dtype=np.float32)
        E[0] = Ex0 * np.cos(phase)
        E[1] = Ey0 * np.cos(phase)
        E[2] = Ez0 * np.cos(phase)
        
        return E
    
    @staticmethod
    def generate_wave_superposition(grid: CoordinateGrid3D, 
                                  num_waves: int = config.NUM_WAVES) -> Tuple[np.ndarray, Dict]:
        """
        Generate superposition of multiple plane waves as per paper:
        E = Σ_{i=0}^{n} E0,i exp[-j k_i (x cosφ_i sinθ_i + y sinφ_i sinθ_i + z cosθ_i)]
        
        But with FIXED theta and phi for all waves in the sample.
        """
        # Generate parameters for this sample
        params = PlaneWaveGenerator3D.generate_wave_parameters(num_waves)
        
        # Initialize total E-field
        E_total = np.zeros((3, config.Nx, config.Ny, config.Nz), dtype=np.float32)
        
        # Superposition of waves
        for i in range(num_waves):
            E_wave = PlaneWaveGenerator3D.generate_single_wave(
                grid, params['theta'], params['phi'],
                params['k_values'][i], params['Ex0'][i], params['Ey0'][i]
            )
            E_total += E_wave
        
        return E_total, params

def compute_curl_for_superposition(grid: CoordinateGrid3D, params: Dict) -> np.ndarray:
    """
    TRUE analytic curl using k × E identity.
    """
    theta = params['theta']
    phi = params['phi']
    k_values = params['k_values']
    Ex0 = params['Ex0']
    Ey0 = params['Ey0']
    num_waves = len(k_values)

    curl_total = np.zeros((3, config.Nx, config.Ny, config.Nz), dtype=np.float32)

    for i in range(num_waves):
        k = k_values[i]

        kx = k * np.sin(theta) * np.cos(phi)
        ky = k * np.sin(theta) * np.sin(phi)
        kz = k * np.cos(theta)

        # Compute Ez from divergence-free condition
        if np.abs(np.cos(theta)) > 1e-10:
            Ez0 = -(np.cos(phi) * np.sin(theta) * Ex0[i] +
                    np.sin(phi) * np.sin(theta) * Ey0[i]) / np.cos(theta)
        else:
            Ez0 = 0.0

        # Generate E_i on the SAME grid
        E_i = PlaneWaveGenerator3D.generate_single_wave(
            grid, theta, phi, k, Ex0[i], Ey0[i]
        )

        Ex, Ey, Ez = E_i[0], E_i[1], E_i[2]

        # k × E
        curl_total[0] += ky * Ez - kz * Ey
        curl_total[1] += kz * Ex - kx * Ez
        curl_total[2] += kx * Ey - ky * Ex

    return curl_total


# ============================================================================
# NORMALIZATION
# ============================================================================

class LocalMaxNormalizer3D:
    """Normalization by local maximum for each component as per paper"""
    
    @staticmethod
    def normalize(field: np.ndarray) -> np.ndarray:
        """
        Normalize each component by its local maximum.
        
        Args:
            field: [3, Nx, Ny, Nz] or [B, 3, Nx, Ny, Nz] field to normalize
            
        Returns:
            normalized_field: Normalized field
        """
        if field.ndim == 4:  # [3, Nx, Ny, Nz]
            max_vals = np.max(np.abs(field), axis=(1, 2, 3), keepdims=True)
            max_vals[max_vals == 0.0] = 1.0  # numerical safety
            return field / max_vals
        else:  # [B, 3, Nx, Ny, Nz]
            max_vals = np.max(np.abs(field), axis=(2, 3, 4), keepdims=True)
            max_vals[max_vals == 0.0] = 1.0  # numerical safety
            return field / max_vals

# ============================================================================
# DATASET CLASS
# ============================================================================

class DCODataset3D(Dataset):
    """3D Dataset for E-field to curl(E) mapping"""
    
    def __init__(self, num_samples: int = config.DATASET_SIZE, mode: str = 'train'):
        self.mode = mode
        self.num_samples = num_samples
        self.samples = []
        self.wave_params = []
        
        print(f"Generating {num_samples} {mode} samples...")
        
        for i in range(num_samples):
            # Random cell dimensions (0.3-0.8 mm)
            delta_x = np.random.uniform(config.DELTA_MIN, config.DELTA_MAX)
            delta_y = np.random.uniform(config.DELTA_MIN, config.DELTA_MAX)
            delta_z = np.random.uniform(config.DELTA_MIN, config.DELTA_MAX)
            
            # Create grid with these cell dimensions
            grid = CoordinateGrid3D(delta_x, delta_y, delta_z)
            
            # Generate E-field superposition and parameters
            E_field, params = PlaneWaveGenerator3D.generate_wave_superposition(grid)
            
            # Compute analytic curl
            curl_gt = compute_curl_for_superposition(grid, params)
            
            # Normalize curl by local maximum (as per paper, target only)
            curl_norm = LocalMaxNormalizer3D.normalize(curl_gt)
            
            # Get coordinate deltas (trunk input)
            # coords = grid.get_coordinate_deltas()
            coords = grid.get_trunk_input()
            
            # Store sample
            self.samples.append({
                'E_field': E_field.astype(np.float32),
                'coords': coords,
                'curl_norm': curl_norm.astype(np.float32),
                'curl_gt': curl_gt.astype(np.float32),
                'delta': (delta_x, delta_y, delta_z)
            })
            self.wave_params.append(params)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample['E_field']),      # Branch input
            torch.from_numpy(sample['coords']),       # Trunk input
            torch.from_numpy(sample['curl_norm']),    # Normalized target
            torch.from_numpy(sample['curl_gt'])       # Ground truth (for metrics)
        )
    
    def get_sample_info(self, idx):
        """Get sample information for analysis"""
        return {
            'wave_params': self.wave_params[idx],
            'delta': self.samples[idx]['delta']
        }

# ============================================================================
# MODEL ARCHITECTURE: DCO-UNET 3D
# ============================================================================

class ResidualConvBlock3D(nn.Module):
    """3×3×3 residual convolutional block with GELU activation"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Main path: Conv3D -> GELU -> Conv3D -> GELU
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              padding=1, stride=1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              padding=1, stride=1)
        self.gelu2 = nn.GELU()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.gelu1(self.conv1(x))
        out = self.gelu2(self.conv2(out))
        return out + identity

class DownsamplingBranch(nn.Module):
    """Downsampling branch with max-pooling (2×2×2) between blocks"""
    
    def __init__(self, in_channels: int, init_channels: int = 32, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        
        # First convolution
        self.init_conv = ResidualConvBlock3D(in_channels, init_channels)
        
        # Downsampling blocks
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = init_channels
        for i in range(num_levels - 1):
            # Downsample
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            
            # Double channels
            channels *= 2
            self.blocks.append(ResidualConvBlock3D(channels // 2, channels))
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns skip connections and bottleneck"""
        skips = []
        
        # Initial block
        x = self.init_conv(x)
        skips.append(x)
        
        # Downsampling levels
        for pool, block in zip(self.pools, self.blocks):
            x = pool(x)      # Downsample
            x = block(x)     # Process
            skips.append(x)  # Save for skip connection
        
        return skips[:-1], x  # Last is bottleneck, not skip

class UpsamplingBranch(nn.Module):
    """Upsampling branch with transposed convolutions (2×2×2)"""
    
    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        channels = in_channels
        for i in range(num_levels - 1):
            # Upsample (halve channels)
            self.upconvs.append(
                nn.ConvTranspose3d(channels, channels // 2, kernel_size=2, stride=2)
            )
            channels //= 2
            
            # Process block (input channels doubled due to skip connection)
            self.blocks.append(ResidualConvBlock3D(channels * 2, channels))
        
        # Final convolution to output channels
        self.final_conv = nn.Conv3d(channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Upsample with skip connections"""
        # Process from bottleneck
        for i, (upconv, block) in enumerate(zip(self.upconvs, self.blocks)):
            x = upconv(x)  # Upsample
            
            # Skip connection (corresponding level from downsampling)
            skip = skips[-(i + 1)]
            
            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)
            
            # Process
            x = block(x)
        
        return self.final_conv(x)

class DCO_UNet3D(nn.Module):
    """
    DCO-UNet 3D: Differentiable Curl Operator Network
    
    Architecture:
    - Two downsampling branches (branch for E-field, trunk for coordinates)
    - Hadamard product at bottleneck
    - Upsampling branch with skip connections from E-field branch
    """
    
    def __init__(self, 
                 branch_in_channels: int = 3,  # E-field components
                 trunk_in_channels: int = 3,   # Coordinate deltas
                 out_channels: int = 3,        # Curl components
                 init_channels: int = 32,
                 num_levels: int = 4):
        super().__init__()
        
        # 1. Branch network (E-field processing)
        self.branch_down = DownsamplingBranch(branch_in_channels, init_channels, num_levels)
        
        # 2. Trunk network (coordinate processing)
        self.trunk_down = DownsamplingBranch(trunk_in_channels, init_channels, num_levels)
        
        # 3. Bottleneck processing (after Hadamard product)
        bottleneck_channels = init_channels * (2 ** (num_levels - 1))
        self.bottleneck = ResidualConvBlock3D(bottleneck_channels, bottleneck_channels)
        
        # 4. Upsampling network
        self.up = UpsamplingBranch(bottleneck_channels, out_channels, num_levels)
        
    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DCO-UNet.
        
        Args:
            branch_input: [B, 3, Nx, Ny, Nz] - E-field
            trunk_input:  [B, 3, Nx, Ny, Nz] - Coordinate deltas
            
        Returns:
            Output: [B, 3, Nx, Ny, Nz] - Predicted normalized curl
        """
        # 1. Process through downsampling branches
        branch_skips, branch_bottleneck = self.branch_down(branch_input)
        # Trunk network only provides bottleneck features (no skip connections used)
        _, trunk_bottleneck = self.trunk_down(trunk_input)
        
        # 2. Hadamard product at bottleneck (element-wise multiplication)
        combined = branch_bottleneck * trunk_bottleneck
        
        # 3. Process through bottleneck block
        x = self.bottleneck(combined)
        
        # 4. Upsample with skip connections from branch only
        output = self.up(x, branch_skips)
        
        return output

# ============================================================================
# METRICS CALCULATION
# ============================================================================

class MetricsCalculator3D:
    """Calculate metrics including MRE as per paper"""
    
    @staticmethod
    def calculate_mre(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
        """
        Calculate Mean Relative Error (MRE) as per paper formula.
        
        MRE = (1/Nx Ny Nz) Σ_{i,j,k=1}^{Nx,Ny,Nz} 
              (y_{i,j,k}^p - y_{i,j,k}^t) / y_{i,j,k}^t  if y_{i,j,k}^t ≠ 0
              y_{i,j,k}^p if y_{i,j,k}^t = 0
              
        where y^t = ground truth, y^p = prediction
        """
        Nx, Ny, Nz = pred.shape[1:]
        total_cells = Nx * Ny * Nz
        
        mre_sum = 0.0
        num_components = pred.shape[0]
        
        for c in range(num_components):
            pred_c = pred[c].flatten()
            target_c = target[c].flatten()
            
            for i in range(len(pred_c)):
                if np.abs(target_c[i]) > eps:
                    mre_sum += np.abs(pred_c[i] - target_c[i]) / np.abs(target_c[i])
                else:
                    mre_sum += np.abs(pred_c[i])
        
        return mre_sum / (num_components * total_cells)
    
    @staticmethod
    def calculate_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate MSE for each component"""
        mse = np.zeros(pred.shape[0])
        for c in range(pred.shape[0]):
            mse[c] = np.mean((pred[c] - target[c])**2)
        return mse
    
    @staticmethod
    def calculate_normalized_mse(pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate normalized MSE"""
        return np.mean((pred - target)**2)

# ============================================================================
# TRAINING LOOP WITH PROPER PRINTING
# ============================================================================

class DCO_Trainer:
    """Handles model training and evaluation with epoch-wise printing"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'epoch': []
        }
    
    def compute_loss(self, loader: DataLoader, mode: str = 'val') -> float:
        """Compute loss for a given data loader"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                branch_input, trunk_input, targets, _ = batch
                
                branch_input = branch_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(branch_input, trunk_input)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch without batch printing"""
        self.model.train()
        total_loss = 0
        total_batches = 0
        
        for batch in train_loader:
            branch_input, trunk_input, targets, _ = batch
            
            branch_input = branch_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(branch_input, trunk_input)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
            test_loader: DataLoader, epochs: int = config.EPOCHS):
        """Main training loop with clean epoch-wise printing"""
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        
        # Header for epoch progress
        print(f"\n{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Test Loss':^12} | {'Status':^10}")
        print(f"{'-'*7}+{'-'*14}+{'-'*14}+{'-'*14}+{'-'*12}")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.compute_loss(val_loader, 'val')
            
            # Test
            test_loss = self.compute_loss(test_loader, 'test')
            
            # Save history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['test_loss'].append(test_loss)
            
            # Format losses for display
            train_str = f"{train_loss:.6f}"
            val_str = f"{val_loss:.6f}"
            test_str = f"{test_loss:.6f}"
            
            # Highlight best validation loss
            status = " "
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                status = "BEST VAL"
            
            # Print epoch results
            print(f"{epoch+1:^7d} | {train_str:^12} | {val_str:^12} | {test_str:^12} | {status:^10}")
            
            # Save best model
            if val_loss < best_val_loss:
                self.save_model('best_model.pth')
        
        # Save final model
        self.save_model('final_model.pth')
        return self.history    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config.__dict__
        }
        
        config.MODEL_DIR.mkdir(exist_ok=True)
        torch.save(checkpoint, config.MODEL_DIR / filename)
        print(f"Model saved to {config.MODEL_DIR / filename}")

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualization3D:
    """Create 3D visualizations for results"""
    
    @staticmethod
    def plot_efield_slices(E: np.ndarray, title: str = "E-field Components"):
        """Plot E-field slices"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        components = ['Ex', 'Ey', 'Ez']
        slice_idx = config.Nx // 2
        
        vmax = np.max(np.abs(E))
        
        for i, (ax, comp) in enumerate(zip(axes, components)):
            im = ax.imshow(E[i, :, :, slice_idx], cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
            ax.set_title(f'{comp} (z={slice_idx})')
            ax.set_xlabel('X index')
            ax.set_ylabel('Y index')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_curl_comparison(curl_analytical: np.ndarray, curl_predicted: np.ndarray,
                            title: str = "Curl Comparison"):
        """Plot analytical vs predicted curl"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        components = ['Curl_x', 'Curl_y', 'Curl_z']
        slice_idx = config.Nx // 2
        
        vmax = max(np.max(np.abs(curl_analytical)), np.max(np.abs(curl_predicted)))
        
        for i in range(3):
            # Analytical
            im1 = axes[i, 0].imshow(curl_analytical[i, :, :, slice_idx],
                                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, 0].set_title(f'Analytical {components[i]}')
            axes[i, 0].set_xlabel('X index')
            axes[i, 0].set_ylabel('Y index')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Predicted
            im2 = axes[i, 1].imshow(curl_predicted[i, :, :, slice_idx],
                                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, 1].set_title(f'Predicted {components[i]}')
            axes[i, 1].set_xlabel('X index')
            axes[i, 1].set_ylabel('Y index')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Difference
            diff = curl_predicted[i] - curl_analytical[i]
            im3 = axes[i, 2].imshow(diff[:, :, slice_idx], cmap='RdBu_r')
            axes[i, 2].set_title(f'Difference {components[i]}')
            axes[i, 2].set_xlabel('X index')
            axes[i, 2].set_ylabel('Y index')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_mre_heatmap(curl_analytical: np.ndarray, curl_predicted: np.ndarray,
                        mre: float, title: str = "MRE Heatmap"):
        """Plot MRE heatmap"""
        Nx, Ny, Nz = curl_analytical.shape[1:]
        mre_map = np.zeros((Nx, Ny, Nz))
        
        # Calculate MRE at each point
        eps = 1e-8
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    mre_sum = 0
                    for c in range(3):
                        target = curl_analytical[c, i, j, k]
                        pred = curl_predicted[c, i, j, k]
                        
                        if np.abs(target) > eps:
                            mre_sum += np.abs(pred - target) / np.abs(target)
                        else:
                            mre_sum += np.abs(pred)
                    
                    mre_map[i, j, k] = mre_sum / 3
        
        # Plot slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slice_idx = config.Nx // 2
        
        slices = [
            (mre_map[:, :, slice_idx], 'XY Slice', 'X', 'Y'),
            (mre_map[:, slice_idx, :], 'XZ Slice', 'X', 'Z'),
            (mre_map[slice_idx, :, :], 'YZ Slice', 'Y', 'Z')
        ]
        
        for ax, (slice_data, title_slice, xlabel, ylabel) in zip(axes, slices):
            im = ax.imshow(slice_data, cmap='hot_r', aspect='auto')
            ax.set_title(f'{title_slice}')
            ax.set_xlabel(f'{xlabel} index')
            ax.set_ylabel(f'{ylabel} index')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'{title}\nOverall MRE: {mre:.6f}')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_difference_heatmap(curl_analytical: np.ndarray, curl_predicted: np.ndarray,
                               title: str = "Absolute Difference Heatmap"):
        """Plot absolute difference heatmap"""
        diff = np.abs(curl_predicted - curl_analytical)
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=0))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slice_idx = config.Nx // 2
        
        slices = [
            (diff_magnitude[:, :, slice_idx], 'XY Slice', 'X', 'Y'),
            (diff_magnitude[:, slice_idx, :], 'XZ Slice', 'X', 'Z'),
            (diff_magnitude[slice_idx, :, :], 'YZ Slice', 'Y', 'Z')
        ]
        
        for ax, (slice_data, title_slice, xlabel, ylabel) in zip(axes, slices):
            im = ax.imshow(slice_data, cmap='viridis', aspect='auto')
            ax.set_title(f'{title_slice}')
            ax.set_xlabel(f'{xlabel} index')
            ax.set_ylabel(f'{ylabel} index')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_progress(self):
        """Plot training progress after completion"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = self.history['epoch']
        
        # Plot linear scale
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].plot(epochs, self.history['test_loss'], 'g-', label='Test Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot log scale
        axes[1].semilogy(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[1].semilogy(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[1].semilogy(epochs, self.history['test_loss'], 'g-', label='Test Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss (log scale)')
        axes[1].set_title('Training Progress (Log Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        config.FIGURES_DIR.mkdir(exist_ok=True)
        plt.savefig(config.FIGURES_DIR / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine3D:
    """Handles model inference for test cases"""
    
    def __init__(self, model_path: str = 'models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = DCO_UNet3D().to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found")
    
    def generate_test_case(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Generate test case as per paper:
        - delta x = delta y = delta z = 0.6 mm
        - θ = 45°, φ = 60°
        - k: 20 values uniformly selected from 0.021 to 838.34
          (frequency spectrum from 1 MHz to 40 GHz)
        """
        # Create grid with test delta
        grid = CoordinateGrid3D(config.INFERENCE_DELTA, 
                               config.INFERENCE_DELTA, 
                               config.INFERENCE_DELTA)
        
        # Generate k values uniformly
        k_values = np.linspace(config.INFERENCE_K_MIN, 
                              config.INFERENCE_K_MAX, 
                              config.INFERENCE_K_VALUES)
        
        # Initialize E-field
        E_total = np.zeros((3, config.Nx, config.Ny, config.Nz), dtype=np.float32)
        
        # Prepare parameters for curl computation
        params = {
            'theta': config.INFERENCE_THETA,
            'phi': config.INFERENCE_PHI,
            'k_values': k_values,
            'Ex0': np.random.uniform(*config.AMPLITUDE_RANGE, config.INFERENCE_K_VALUES),
            'Ey0': np.random.uniform(*config.AMPLITUDE_RANGE, config.INFERENCE_K_VALUES)
        }
        
        # Superposition of waves with different k but same theta, phi
        for i, k in enumerate(k_values):
            # Random amplitudes (0-5) for Ex, Ey
            Ex0 = params['Ex0'][i]
            Ey0 = params['Ey0'][i]
            
            # Generate wave
            wave = PlaneWaveGenerator3D.generate_single_wave(
                grid, config.INFERENCE_THETA, config.INFERENCE_PHI, k, Ex0, Ey0
            )
            E_total += wave
        
        # Compute analytical curl
        curl_analytical = compute_curl_for_superposition(grid, params)
        
        # Get coordinate deltas
        coords = grid.get_coordinate_deltas()
        
        # Test case information
        test_info = {
            'delta_mm': config.INFERENCE_DELTA * 1000,
            'theta_deg': np.rad2deg(config.INFERENCE_THETA),
            'phi_deg': np.rad2deg(config.INFERENCE_PHI),
            'k_values': k_values.tolist(),
            'freq_min_MHz': k_values.min() * config.c0 / (2 * np.pi) / 1e6,
            'freq_max_GHz': k_values.max() * config.c0 / (2 * np.pi) / 1e9
        }
        
        return E_total, coords, curl_analytical, test_info
    
    def predict(self, E_field: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Run model prediction - outputs normalized curl directly"""
        # Prepare inputs
        E_tensor = torch.from_numpy(E_field).unsqueeze(0).to(self.device)
        coords_tensor = torch.from_numpy(coords).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Model outputs normalized curl (as per paper)
            curl_pred_norm = self.model(E_tensor, coords_tensor)
            curl_pred_norm = curl_pred_norm.squeeze().cpu().numpy()
        
        # Return normalized prediction (model already outputs normalized curl)
        return curl_pred_norm

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("DCO-UNet 3D: Differentiable Curl Operator Learning")
    print("Based on: 'A Differentiable Curl Operator for Deep Learning Electromagnetics'")
    print("="*70)
    
    # Create directories
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    # 1. Generate dataset
    print("\n1. Generating dataset...")
    dataset = DCODataset3D(num_samples=config.DATASET_SIZE)
    
    # Split dataset (80/10/10)
    train_size = int(config.DATASET_SIZE * config.TRAIN_RATIO)
    val_size = int(config.DATASET_SIZE * config.VAL_RATIO)
    test_size = config.DATASET_SIZE - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # 2. Initialize model
    print("\n2. Initializing model...")
    model = DCO_UNet3D(
        branch_in_channels=3,
        trunk_in_channels=3,
        out_channels=3,
        init_channels=config.INIT_CHANNELS,
        num_levels=config.NUM_LEVELS
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Train model
    print("\n3. Training model...")
    trainer = DCO_Trainer(model, config)
    history = trainer.train(train_loader, val_loader, test_loader, epochs=config.EPOCHS)
    
    # 4. Plot training history
    print("\n4. Plotting training history...")
    viz = Visualization3D()
    fig = viz.plot_training_history(history)
    plt.savefig(config.FIGURES_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Run inference on test case
    print("\n5. Running inference on test case...")
    inference_engine = InferenceEngine3D('models/best_model.pth')
    
    # Generate test case
    E_field, coords, curl_analytical, test_info = inference_engine.generate_test_case()
    
    # Normalize analytical curl for comparison (same as training target)
    curl_analytical_norm = LocalMaxNormalizer3D.normalize(curl_analytical)
    
    # Run prediction (model outputs normalized curl)
    curl_pred_norm = inference_engine.predict(E_field, coords)
    
    # Calculate metrics in normalized space
    mre = MetricsCalculator3D.calculate_mre(curl_pred_norm, curl_analytical_norm)
    mse = MetricsCalculator3D.calculate_mse(curl_pred_norm, curl_analytical_norm)
    nmse = MetricsCalculator3D.calculate_normalized_mse(curl_pred_norm, curl_analytical_norm)
    
    print(f"\nTest Case Results (in normalized space):")
    print(f"  Mean Relative Error (MRE): {mre:.6f}")
    print(f"  Normalized MSE: {nmse:.6e}")
    print(f"  Component MSE:")
    print(f"    X: {mse[0]:.6e}")
    print(f"    Y: {mse[1]:.6e}")
    print(f"    Z: {mse[2]:.6e}")
    
    print(f"\nTest Case Parameters:")
    print(f"  Δx = Δy = Δz = {test_info['delta_mm']:.1f} mm")
    print(f"  θ = {test_info['theta_deg']:.1f}°")
    print(f"  φ = {test_info['phi_deg']:.1f}°")
    print(f"  Frequency range: {test_info['freq_min_MHz']:.1f} MHz to {test_info['freq_max_GHz']:.2f} GHz")
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    
    # Plot input E-field
    fig = viz.plot_efield_slices(E_field, 
                                f"Input E-field: θ={test_info['theta_deg']:.1f}°, "
                                f"φ={test_info['phi_deg']:.1f}°, "
                                f"Δ={test_info['delta_mm']:.1f} mm")
    plt.savefig(config.FIGURES_DIR / 'input_e_field.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot curl comparison (using normalized values as per paper)
    fig = viz.plot_curl_comparison(curl_analytical_norm, curl_pred_norm,
                                  "Curl Comparison: Analytical vs Predicted (Normalized)")
    plt.savefig(config.FIGURES_DIR / 'curl_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot MRE heatmap
    fig = viz.plot_mre_heatmap(curl_analytical_norm, curl_pred_norm, mre,
                              f"MRE Heatmap (MRE = {mre:.4f})")
    plt.savefig(config.FIGURES_DIR / 'mre_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot difference heatmap
    fig = viz.plot_difference_heatmap(curl_analytical_norm, curl_pred_norm,
                                     "Absolute Difference Heatmap (Normalized)")
    plt.savefig(config.FIGURES_DIR / 'difference_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Save results
    print("\n7. Saving results...")
    results = {
        'test_case': test_info,
        'metrics': {
            'mre': float(mre),
            'normalized_mse': float(nmse),
            'mse_components': mse.tolist()
        },
        'training': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_test_loss': history['test_loss'][-1],
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE
        }
    }
    
    with open(config.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("All tasks completed successfully!")
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print(f"Models saved to: {config.MODEL_DIR}")
    print(f"Figures saved to: {config.FIGURES_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
    # Plot training progress
    Visualization3D.plot_training_progress()

    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS:")
    print("="*70)
    print(f"Best validation loss: {min(Visualization3D.history['val_loss']):.6f}")
    print(f"Test loss at best validation: {Visualization3D.history['test_loss'][Visualization3D.history['val_loss'].index(min(Visualization3D.history['val_loss']))]:.6f}")
    print(f"Final training loss: {Visualization3D.history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {Visualization3D.history['val_loss'][-1]:.6f}")
    print(f"Final test loss: {Visualization3D.history['test_loss'][-1]:.6f}")
    print("="*70)