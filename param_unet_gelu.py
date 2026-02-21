"""
DCO-UNet 3D: Differentiable Curl Operator via Deep Learning
Physics-informed 3D convolutional neural network for learning curl operator from E-field

Complete implementation following paper specifications exactly:
- Modified U-net with trunk MLP network processing only cell-size information
- 3D convolutions with residual connections and GELU activation
- Superposition of 20 plane waves with analytical curl computation
- Strict 80/20 train-test split (no validation)
- MSE during training on normalized curl, MRE for inference only
- Hadamard product only at bottleneck
- Proper skip connections between branch downsampling and upsampling
- Cell sizes (Δx, Δy, Δz) as trunk input (passed as 1D features)
- Normalize ground truth curl by local maximum before computing MSE during training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Any
import warnings
import sys
import time
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config:
    """Configuration parameters for the entire system"""
    # Physical constants
    c0: float = 3e8  # Speed of light (m/s)
    pi: float = np.pi
    
    # Grid parameters (32x32x32 as per paper)
    Nx: int = 32
    Ny: int = 32
    Nz: int = 32
    N_CELLS: int = 32
    
    # Cell dimensions range (0.3-0.8 mm as per paper)
    DELTA_MIN: float = 0.3e-3  # 0.3 mm
    DELTA_MAX: float = 0.8e-3  # 0.8 mm
    
    # Wave parameters
    NUM_WAVES: int = 20  # n=20 superimposed waves
    THETA_RANGE: Tuple[float, float] = (0, np.pi)
    PHI_RANGE: Tuple[float, float] = (-np.pi, np.pi)
    K_MIN: float = 0
    K_MAX: float = 1048
    AMPLITUDE_RANGE: Tuple[float, float] = (0, 5)
    
    # Dataset parameters
    DATASET_SIZE: int = 1000
    TRAIN_RATIO: float = 0.8
    BATCH_SIZE: int = 32
    
    # Training parameters
    EPOCHS: int = 300
    LEARNING_RATE: float = 1e-4
    
    # Model parameters
    INIT_CHANNELS: int = 16 # number of channels in the initial first layer
    NUM_LEVELS: int = 4
    DROPOUT_RATE: float = 0.1

    # Output directories
    OUTPUT_DIR: Path = Path("outputs")
    MODEL_DIR: Path = Path("models")
    FIGURES_DIR: Path = Path("figures")
    
    # Inference test case
    INFERENCE_DELTA: float = 0.6e-3
    INFERENCE_THETA: float = np.deg2rad(45)
    INFERENCE_PHI: float = np.deg2rad(60)
    INFERENCE_K_VALUES: int = 20
    INFERENCE_K_MIN: float = 0.021
    INFERENCE_K_MAX: float = 838.34
    
    def __post_init__(self):
        """Initialize directories after dataclass creation"""
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.MODEL_DIR.mkdir(exist_ok=True, parents=True)
        self.FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# ========================================================================================================================================================
# COORDINATE GRID GENERATION
# ========================================================================================================================================================

class CoordinateGrid3D:
    """Manages 3D coordinate grid generation with variable cell sizes"""
    
    def __init__(self, delta_x: float, delta_y: float, delta_z: float):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
        
        # Generate coordinates at cell centers (N_CELLS cells)
        self.x = np.linspace(delta_x/2, Config.N_CELLS * delta_x - delta_x/2, Config.Nx)
        self.y = np.linspace(delta_y/2, Config.N_CELLS * delta_y - delta_y/2, Config.Ny)
        self.z = np.linspace(delta_z/2, Config.N_CELLS * delta_z - delta_z/2, Config.Nz)
        
        # Create meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def get_cell_size_tensor(self) -> np.ndarray:
        """
        Get cell size tensor for trunk network.
        Now simply returns the 3 scalars instead of a full 3D grid, to be processed by the MLP.
        
        Returns:
            cell_sizes: [3,] array (delta_x, delta_y, delta_z)
        """
        return np.array([self.delta_x, self.delta_y, self.delta_z], dtype=np.float32)

# ========================================================================================================================================================
# NORMALIZATION UTILITIES
# ========================================================================================================================================================

class LocalMaxNormalizer3D:
    """Normalize curl by local maximum for each component"""
    
    @staticmethod
    def normalize_with_scale(curl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        curl_norm = np.zeros_like(curl, dtype=np.float32)
        scale = np.zeros(3, dtype=np.float32)
        
        for i in range(3):
            max_val = np.max(np.abs(curl[i]))
            if max_val > 1e-8:
                curl_norm[i] = curl[i] / max_val
                scale[i] = max_val
            else:
                curl_norm[i] = curl[i]
                scale[i] = 1.0
        
        return curl_norm, scale
    
    @staticmethod
    def normalize_tensor_with_scale(curl_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = curl_tensor.shape[0]
        curl_norm = torch.zeros_like(curl_tensor)
        scale = torch.zeros(batch_size, 3, device=curl_tensor.device)
        
        for b in range(batch_size):
            for i in range(3):
                max_val = torch.max(torch.abs(curl_tensor[b, i]))
                if max_val > 1e-8:
                    curl_norm[b, i] = curl_tensor[b, i] / max_val
                    scale[b, i] = max_val
                else:
                    curl_norm[b, i] = curl_tensor[b, i]
                    scale[b, i] = 1.0
        
        return curl_norm, scale

# ========================================================================================================================================================
# PHYSICS: PLANE WAVE GENERATION
# ========================================================================================================================================================

class PlaneWaveGenerator3D:
    """Generates plane wave superpositions as per paper specifications"""
    
    @staticmethod
    def generate_wave_parameters(num_waves: int = Config.NUM_WAVES) -> Dict[str, Any]:
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(-np.pi, np.pi)
        k_values = np.random.uniform(Config.K_MIN, Config.K_MAX, num_waves)
        Ex0_values = np.random.uniform(*Config.AMPLITUDE_RANGE, num_waves)
        Ey0_values = np.random.uniform(*Config.AMPLITUDE_RANGE, num_waves)
        
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
        kx = k * np.sin(theta) * np.cos(phi)
        ky = k * np.sin(theta) * np.sin(phi)
        kz = k * np.cos(theta)
        
        if np.abs(kz) > 1e-10:
            Ez0 = -(kx * Ex0 + ky * Ey0) / kz
        else:
            Ez0 = 0.0
        
        phase = kx * grid.X + ky * grid.Y + kz * grid.Z
        
        E = np.zeros((3, Config.Nx, Config.Ny, Config.Nz), dtype=np.float32)
        E[0] = Ex0 * np.cos(phase)
        E[1] = Ey0 * np.cos(phase)
        E[2] = Ez0 * np.cos(phase)
        
        return E
    
    @staticmethod
    def generate_wave_superposition(grid: CoordinateGrid3D, 
                                  num_waves: int = Config.NUM_WAVES) -> Tuple[np.ndarray, Dict[str, Any]]:
        params = PlaneWaveGenerator3D.generate_wave_parameters(num_waves)
        E_total = np.zeros((3, Config.Nx, Config.Ny, Config.Nz), dtype=np.float32)
        
        for i in range(num_waves):
            E_wave = PlaneWaveGenerator3D.generate_single_wave(
                grid, params['theta'], params['phi'],
                params['k_values'][i], params['Ex0'][i], params['Ey0'][i]
            )
            E_total += E_wave
        
        return E_total, params

# ========================================================================================================================================================
# ANALYTICAL CURL COMPUTATION
# ========================================================================================================================================================

class AnalyticalCurlCalculator:
    """Computes analytic curl for wave superpositions using k × E identity"""
    
    @staticmethod
    def compute_curl_for_superposition(grid: CoordinateGrid3D, params: Dict[str, Any]) -> np.ndarray:
        theta = params['theta']
        phi = params['phi']
        k_values = params['k_values']
        Ex0 = params['Ex0']
        Ey0 = params['Ey0']
        num_waves = len(k_values)
        
        curl_total = np.zeros((3, Config.Nx, Config.Ny, Config.Nz), dtype=np.float32)
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        for i in range(num_waves):
            k = k_values[i]
            
            kx = k * sin_theta * cos_phi
            ky = k * sin_theta * sin_phi
            kz = k * cos_theta
            
            if np.abs(kz) > 1e-10:
                Ez0 = -(kx * Ex0[i] + ky * Ey0[i]) / kz
            else:
                Ez0 = 0.0
            
            phase = kx * grid.X + ky * grid.Y + kz * grid.Z
            sin_phase = np.sin(phase)
            
            curl_x = ky * Ez0 - kz * Ey0[i]
            curl_y = kz * Ex0[i] - kx * Ez0
            curl_z = kx * Ey0[i] - ky * Ex0[i]
            
            curl_total[0] -= curl_x * sin_phase
            curl_total[1] -= curl_y * sin_phase
            curl_total[2] -= curl_z * sin_phase
        
        return curl_total

# ========================================================================================================================================================
# DCO-UNet 3D ARCHITECTURE
# ========================================================================================================================================================

class RowdyActivation(nn.Module):
    """
    Strict implementation of Rowdy Activation from RiemannONet Paper (Eq. 10).
    g(x) = tanh(10a*x + c) + 10a1 * sin(10F1*x + c1)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.n = 10.0 # Scaling factor defined in paper
        
        self.a = nn.Parameter(torch.full((1, in_channels, 1, 1, 1), 0.1))
        self.c = nn.Parameter(torch.full((1, in_channels, 1, 1, 1), 0.1))
        self.a1 = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
        self.c1 = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
        self.F1 = nn.Parameter(torch.full((1, in_channels, 1, 1, 1), 0.1))

    def forward(self, x):
        base = torch.tanh(self.n * self.a * x + self.c)
        rowdy = (self.n * self.a1) * torch.sin(self.n * self.F1 * x + self.c1)
        return base + rowdy

class ResidualBlock3D(nn.Module):
    """Encoder/Decoder Conv Block (Green/Grey Blocks in Fig 2)."""
    def __init__(self, in_channels, out_channels, use_rowdy=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act1 = RowdyActivation(out_channels) if use_rowdy else nn.GELU()
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act2 = RowdyActivation(out_channels) if use_rowdy else nn.GELU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.gn2(x)
        
        x = x + residual
        x = self.act2(x)
        return x

class UpsampleBlock(nn.Module):
    """Decoder Upsample Block (Yellow Block in Fig 2)."""
    def __init__(self, in_channels, out_channels, use_rowdy=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act = RowdyActivation(out_channels) if use_rowdy else nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.up(x)))

# ============================================================================
# NEW TRUNK MLP ARCHITECTURE
# ============================================================================

class MLPBlock(nn.Module):
    """Individual MLP Block (Dense -> BatchNorm -> Swish -> Dropout)"""
    def __init__(self, in_features, out_features, dropout_rate=0.1, use_dropout=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.SiLU() # SiLU is Swish activation
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class TrunkMLP(nn.Module):
    """MLP Network processing the 1D geometry parameters"""
    def __init__(self, in_features=3, dropout_rate=0.1):
        super().__init__()
        # Matches dimensions and specifications in diagram
        self.block1 = MLPBlock(in_features, 16, dropout_rate, use_dropout=True)
        self.block2 = MLPBlock(16, 32, dropout_rate, use_dropout=True)
        self.block3 = MLPBlock(32, 64, dropout_rate, use_dropout=False)
        self.block4 = MLPBlock(64, 128, dropout_rate, use_dropout=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class ProjectionFusionBlock(nn.Module):
    """
    Projection layer + Element-wise multiplication for conditioning.
    Dynamically projects the 128-dim trunk output to match the branch's channel sizes.
    """
    def __init__(self, mlp_out_features, branch_channels):
        super().__init__()
        self.projection = nn.Linear(mlp_out_features, branch_channels)

    def forward(self, x_branch, x_trunk_mlp_out):
        # x_branch: [B, C, D, H, W]
        # x_trunk_mlp_out: [B, mlp_out_features]
        proj = self.projection(x_trunk_mlp_out) # [B, C]
        
        # Reshape to [B, C, 1, 1, 1] for element-wise broadcasting
        proj = proj.view(proj.size(0), proj.size(1), 1, 1, 1)
        
        return x_branch * proj

# ============================================================================
# MAIN NETWORK (DCO-UNet 3D)
# ============================================================================

class DCO_UNet3D(nn.Module):
    def __init__(self, branch_in_channels=3, trunk_in_features=3, 
                 out_channels=3, init_channels=32, num_levels=4, 
                 dropout_rate=0.1, use_rowdy=True):
        super().__init__()
        self.num_levels = num_levels
        
        # --- BRANCH ENCODERS ---
        self.branch_encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # --- TRUNK MLP & PROJECTION BLOCKS ---
        self.trunk_mlp = TrunkMLP(in_features=trunk_in_features, dropout_rate=dropout_rate)
        mlp_out_features = 128  # Final output from MLP Block 4 is exactly 128
        self.fusion_blocks = nn.ModuleList()
        
        # Level 1 (Input)
        self.branch_encoders.append(ResidualBlock3D(branch_in_channels, init_channels, use_rowdy))
        self.fusion_blocks.append(ProjectionFusionBlock(mlp_out_features, init_channels))
        
        # Levels 2 to N
        for i in range(1, num_levels):
            in_c = init_channels * (2 ** (i - 1))
            out_c = init_channels * (2 ** i)
            
            # Downsampling (Red Block)
            self.downsamplers.append(nn.MaxPool3d(2))
            
            # Dropout (Blue Block - After MaxPool)
            self.dropouts.append(nn.Dropout3d(p=dropout_rate))
            
            # Conv Blocks (Green Block)
            self.branch_encoders.append(ResidualBlock3D(in_c, out_c, use_rowdy))
            
            # Fusion
            self.fusion_blocks.append(ProjectionFusionBlock(mlp_out_features, out_c))

        # --- DECODER ---
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(num_levels - 1, 0, -1):
            in_c = init_channels * (2 ** i)
            out_c = init_channels * (2 ** (i - 1))
            
            # Yellow Block (Upsample + GN + Act)
            self.upsamplers.append(UpsampleBlock(in_c, out_c, use_rowdy))
            
            # Grey Block (Conv + GN + Act)
            # Input is doubled due to concatenation
            self.decoders.append(ResidualBlock3D(out_c * 2, out_c, use_rowdy))
            
        # Final Convolution Mapping to Output Fields
        self.final_conv = nn.Conv3d(init_channels, out_channels, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, branch_input, trunk_input):
        x_b = branch_input
        
        # Robust handling just in case old 5D tensor grid is passed to trunk
        if trunk_input.dim() == 5:
            x_t = trunk_input[:, :, 0, 0, 0] # Extract the 3 parameters to shape [B, 3]
        else:
            x_t = trunk_input
            
        # Process the trunk network to get 128-dim condition vector
        mlp_out = self.trunk_mlp(x_t)
        
        fused_skips = []
        
        # --- ENCODER PATH ---
        for i in range(self.num_levels):
            # 1. Downsample & Dropout (if not first level)
            if i > 0:
                x_b = self.downsamplers[i-1](x_b)
                x_b = self.dropouts[i-1](x_b)
            
            # 2. Conv Blocks (Main Encoder branch goes straight down)
            x_b = self.branch_encoders[i](x_b)
            
            # 3. Apply Conditional Projection Block onto the isolated Skip path
            fused = self.fusion_blocks[i](x_b, mlp_out)
            fused_skips.append(fused)
            
        # --- DECODER PATH ---
        # Start from the bottom-most fused feature (the bottleneck projection)
        x = fused_skips[-1]
        
        for i in range(len(self.upsamplers)):
            # 1. Upsample (Yellow Block)
            x = self.upsamplers[i](x)
            
            # 2. Get Skip Connection
            skip = fused_skips[-(i + 2)]
            
            # 3. Concatenate
            x = torch.cat([x, skip], dim=1)
            
            # 4. Conv Block (Grey Block)
            x = self.decoders[i](x)
            
        return self.final_conv(x)

# ============================================================================
# DATASET CLASS
# ============================================================================

class DCODataset3D(Dataset):
    def __init__(self, num_samples: int = Config.DATASET_SIZE):
        self.num_samples = num_samples
        self.data = []
        
        print(f"Generating {num_samples} samples...")
        for i in range(num_samples):
            delta_x = np.random.uniform(Config.DELTA_MIN, Config.DELTA_MAX)
            delta_y = np.random.uniform(Config.DELTA_MIN, Config.DELTA_MAX)
            delta_z = np.random.uniform(Config.DELTA_MIN, Config.DELTA_MAX)
            
            grid = CoordinateGrid3D(delta_x, delta_y, delta_z)
            E_field, params = PlaneWaveGenerator3D.generate_wave_superposition(grid)
            curl_analytical = AnalyticalCurlCalculator.compute_curl_for_superposition(grid, params)
            curl_norm, scale = LocalMaxNormalizer3D.normalize_with_scale(curl_analytical)
            
            # Extract only the 3 physical features (Delta dimensions) [3,] array
            cell_sizes = grid.get_cell_size_tensor()
            
            self.data.append({
                'E_field': E_field.astype(np.float32),
                'cell_sizes': cell_sizes.astype(np.float32),
                'curl_analytical': curl_analytical.astype(np.float32),
                'curl_norm': curl_norm.astype(np.float32),
                'scale': scale.astype(np.float32),
                'params': params
            })
            
            if (i + 1) % 100 == 0 or (i + 1) == num_samples:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        print("Dataset generation complete!")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        item = self.data[idx]
        return (
            item['E_field'],
            item['cell_sizes'],
            item['curl_analytical'],
            item['curl_norm'],
            item['scale'],
            item['params']
        )

# ============================================================================
# METRICS CALCULATION
# ============================================================================

class MetricsCalculator3D:
    @staticmethod
    def calculate_mre(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
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
    def calculate_mse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target)**2))
    
    @staticmethod
    def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - target)))

# ============================================================================
# TRAINER CLASS (WITH NORMALIZED LOSS)
# ============================================================================

class DCO_Trainer:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        self.history = {'train_loss': [], 'test_loss': [], 'epoch': []}
        print(f"Using device: {self.device}")
    
    def compute_test_loss(self, test_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                E_field, cell_sizes, _, curl_norm_target, _, _ = batch
                
                E_field = E_field.to(self.device)
                cell_sizes = cell_sizes.to(self.device)
                curl_norm_target = curl_norm_target.to(self.device)
                
                outputs_raw = self.model(E_field, cell_sizes)
                outputs_norm, _ = LocalMaxNormalizer3D.normalize_tensor_with_scale(outputs_raw)
                loss = self.criterion(outputs_norm, curl_norm_target)
                
                total_loss += loss.item()
                total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        for batch in train_loader:
            E_field, cell_sizes, _, curl_norm_target, _, _ = batch
            
            E_field = E_field.to(self.device)
            cell_sizes = cell_sizes.to(self.device)
            curl_norm_target = curl_norm_target.to(self.device)
            
            self.optimizer.zero_grad()
            outputs_raw = self.model(E_field, cell_sizes)
            outputs_norm, _ = LocalMaxNormalizer3D.normalize_tensor_with_scale(outputs_raw)
            
            loss = self.criterion(outputs_norm, curl_norm_target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader, 
              epochs: int = Config.EPOCHS) -> Dict[str, List[float]]:
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Training on NORMALIZED curl (MSE loss)")
        print(f"{'='*60}\n")
        
        best_test_loss = float('inf')
        print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Test Loss':^12} | {'Status':^10}")
        print(f"{'-'*8}+{'-'*14}+{'-'*14}+{'-'*12}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.compute_test_loss(test_loader)
            
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            
            train_str = f"{train_loss:.6f}"
            test_str = f"{test_loss:.6f}"
            
            status = ""
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                status = "BEST TEST"
                self.save_model('best_model.pth')
            
            print(f"{epoch+1:^8d} | {train_str:^12} | {test_str:^12} | {status:^10}")
        
        self.save_model('final_model.pth')
        print(f"\nTraining completed!")
        print(f"Best test loss: {best_test_loss:.6f} (on normalized curl)")
        
        return self.history
    
    def save_model(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, self.config.MODEL_DIR / filename)
        print(f"  Model saved to {self.config.MODEL_DIR / filename}")

# ============================================================================
# VISUALIZATION CLASS
# ============================================================================

class Visualization3D:
    @staticmethod
    def plot_efield_slices(E: np.ndarray, title: str = "E-field Components"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        components = ['Ex', 'Ey', 'Ez']
        slice_idx = Config.Nx // 2
        
        vmax = np.max(np.abs(E))
        
        for i, (ax, comp) in enumerate(zip(axes, components)):
            im = ax.imshow(E[i, :, :, slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
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
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        components = ['Curl_x', 'Curl_y', 'Curl_z']
        slice_idx = Config.Nx // 2
        
        vmax = max(np.max(np.abs(curl_analytical)), np.max(np.abs(curl_predicted)))
        
        for i in range(3):
            im1 = axes[i, 0].imshow(curl_analytical[i, :, :, slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, 0].set_title(f'Analytical {components[i]}')
            axes[i, 0].set_xlabel('X index')
            axes[i, 0].set_ylabel('Y index')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            im2 = axes[i, 1].imshow(curl_predicted[i, :, :, slice_idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, 1].set_title(f'Predicted {components[i]}')
            axes[i, 1].set_xlabel('X index')
            axes[i, 1].set_ylabel('Y index')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            diff = np.abs(curl_predicted[i] - curl_analytical[i])
            im3 = axes[i, 2].imshow(diff[:, :, slice_idx], cmap='hot_r')
            axes[i, 2].set_title(f'Abs Difference {components[i]}')
            axes[i, 2].set_xlabel('X index')
            axes[i, 2].set_ylabel('Y index')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_mre_heatmap(curl_analytical: np.ndarray, curl_predicted: np.ndarray,
                        mre: float = None, title: str = "MRE Heatmap"):
        if mre is None:
            mre = MetricsCalculator3D.calculate_mre(curl_predicted, curl_analytical)
        
        Nx, Ny, Nz = curl_analytical.shape[1:]
        mre_map = np.zeros((Nx, Ny, Nz))
        
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
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slice_idx = Config.Nx // 2
        
        slices = [
            (mre_map[:, :, slice_idx], 'XY Slice', 'X', 'Y'),
            (mre_map[:, slice_idx, :], 'XZ Slice', 'X', 'Z'),
            (mre_map[slice_idx, :, :], 'YZ Slice', 'Y', 'Z')
        ]
        
        vmax = np.max(mre_map)
        for ax, (slice_data, title_slice, xlabel, ylabel) in zip(axes, slices):
            im = ax.imshow(slice_data, cmap='hot_r', aspect='auto', vmin=0, vmax=vmax)
            ax.set_title(f'{title_slice}')
            ax.set_xlabel(f'{xlabel} index')
            ax.set_ylabel(f'{ylabel} index')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'{title}\nOverall MRE: {mre:.6f}')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = history['epoch']
        
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss (Normalized)')
        axes[0].set_title('Training Progress (Normalized Space)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].semilogy(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[1].semilogy(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss (log scale)')
        axes[1].set_title('Training Progress (Log Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine3D:
    def __init__(self, model_path: str = 'models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DCO_UNet3D().to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")
    
    def generate_test_case(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        grid = CoordinateGrid3D(Config.INFERENCE_DELTA, Config.INFERENCE_DELTA, Config.INFERENCE_DELTA)
        k_values = np.linspace(Config.INFERENCE_K_MIN, Config.INFERENCE_K_MAX, Config.INFERENCE_K_VALUES)
        
        params = {
            'theta': Config.INFERENCE_THETA,
            'phi': Config.INFERENCE_PHI,
            'k_values': k_values,
            'Ex0': np.random.uniform(*Config.AMPLITUDE_RANGE, Config.INFERENCE_K_VALUES),
            'Ey0': np.random.uniform(*Config.AMPLITUDE_RANGE, Config.INFERENCE_K_VALUES)
        }
        
        E_total = np.zeros((3, Config.Nx, Config.Ny, Config.Nz), dtype=np.float32)
        for i, k in enumerate(k_values):
            Ex0 = params['Ex0'][i]
            Ey0 = params['Ey0'][i]
            E_wave = PlaneWaveGenerator3D.generate_single_wave(
                grid, Config.INFERENCE_THETA, Config.INFERENCE_PHI, k, Ex0, Ey0
            )
            E_total += E_wave
        
        curl_analytical = AnalyticalCurlCalculator.compute_curl_for_superposition(grid, params)
        _, scale = LocalMaxNormalizer3D.normalize_with_scale(curl_analytical)
        cell_sizes = grid.get_cell_size_tensor()
        
        test_info = {
            'delta_mm': Config.INFERENCE_DELTA * 1000,
            'theta_deg': np.rad2deg(Config.INFERENCE_THETA),
            'phi_deg': np.rad2deg(Config.INFERENCE_PHI),
            'k_values': k_values.tolist(),
            'freq_min_MHz': k_values.min() * Config.c0 / (2 * np.pi) / 1e6,
            'freq_max_GHz': k_values.max() * Config.c0 / (2 * np.pi) / 1e9,
            'scale': scale.tolist()
        }
        
        return E_total, cell_sizes, curl_analytical, scale, test_info
    
    def predict(self, E_field: np.ndarray, cell_sizes: np.ndarray) -> np.ndarray:
        E_tensor = torch.from_numpy(E_field).unsqueeze(0).to(self.device)
        
        # Add batch dim manually as model expects [1, 3] size input tensors
        cell_sizes_tensor = torch.from_numpy(cell_sizes).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            curl_pred = self.model(E_tensor, cell_sizes_tensor)
            curl_pred = curl_pred.squeeze().cpu().numpy()
        
        return curl_pred

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("DCO-UNet 3D: Differentiable Curl Operator Learning")
    print("Implementation strictly following specificed MLP trunk architecture")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    cfg = Config()
    
    print("\n1. Generating dataset...")
    full_dataset = DCODataset3D(num_samples=cfg.DATASET_SIZE)
    
    train_size = int(cfg.DATASET_SIZE * cfg.TRAIN_RATIO)
    test_size = cfg.DATASET_SIZE - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print("\n2. Initializing model...")
    model = DCO_UNet3D(
        branch_in_channels=3,
        trunk_in_features=3, # Renamed to reflect 1D array structure
        out_channels=3,
        init_channels=cfg.INIT_CHANNELS,
        num_levels=cfg.NUM_LEVELS,
        dropout_rate=0.2, 
        use_rowdy=False   
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    print("\n3. Training model (MSE on normalized curl)...")
    trainer = DCO_Trainer(model, cfg)
    history = trainer.train(train_loader, test_loader, epochs=cfg.EPOCHS)
    
    print("\n4. Plotting training history...")
    viz = Visualization3D()
    fig = viz.plot_training_history(history, cfg.FIGURES_DIR / 'training_history.png')
    plt.savefig(cfg.FIGURES_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\n5. Running inference on test case (MRE on raw curl)...")
    inference_engine = InferenceEngine3D(cfg.MODEL_DIR / 'best_model.pth')
    E_field, cell_sizes, curl_analytical, scale, test_info = inference_engine.generate_test_case()
    
    curl_pred_raw = inference_engine.predict(E_field, cell_sizes)
    
    mre = MetricsCalculator3D.calculate_mre(curl_pred_raw, curl_analytical)
    mse = MetricsCalculator3D.calculate_mse(curl_pred_raw, curl_analytical)
    mae = MetricsCalculator3D.calculate_mae(curl_pred_raw, curl_analytical)
    
    print(f"\nTest Case Results (RAW curl):")
    print(f"  MRE: {mre:.6f}")
    print(f"  MSE: {mse:.6e}")
    print(f"  MAE: {mae:.6e}")
    
    print("\n6. Creating visualizations...")
    fig = viz.plot_efield_slices(E_field, f"Input E-field: θ={test_info['theta_deg']:.1f}°, φ={test_info['phi_deg']:.1f}°, Δ={test_info['delta_mm']:.1f} mm")
    plt.savefig(cfg.FIGURES_DIR / 'input_e_field.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    fig = viz.plot_curl_comparison(curl_analytical, curl_pred_raw, "Curl Comparison - RAW (Analytical vs Predicted)")
    plt.savefig(cfg.FIGURES_DIR / 'curl_comparison_raw.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    curl_analytical_norm, _ = LocalMaxNormalizer3D.normalize_with_scale(curl_analytical)
    curl_pred_norm, _ = LocalMaxNormalizer3D.normalize_with_scale(curl_pred_raw)
    
    fig = viz.plot_curl_comparison(curl_analytical_norm, curl_pred_norm, "Curl Comparison - NORMALIZED (Analytical vs Predicted)")
    plt.savefig(cfg.FIGURES_DIR / 'curl_comparison_norm.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    fig = viz.plot_mre_heatmap(curl_analytical, curl_pred_raw, mre, f"MRE Heatmap (MRE={mre:.6f})")
    plt.savefig(cfg.FIGURES_DIR / 'mre_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\n7. Saving results...")
    results = {
        'test_case': test_info,
        'metrics_raw': {'mre': float(mre), 'mse': float(mse), 'mae': float(mae)},
        'training': {
            'final_train_loss': history['train_loss'][-1],
            'final_test_loss': history['test_loss'][-1],
            'best_test_loss': min(history['test_loss']),
            'epochs': cfg.EPOCHS,
            'learning_rate': cfg.LEARNING_RATE
        }
    }
    
    with open(cfg.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
