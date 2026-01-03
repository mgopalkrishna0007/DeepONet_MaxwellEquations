"""
DCO-UNet 3D: Differentiable Curl Operator via Deep Learning
Physics-informed 3D convolutional neural network for learning curl operator from E-field to curl(E)

Based on paper: "A Differentiable Curl Operator for Deep Learning Electromagnetics"

Key Features:
1. Modified U-net with DeepONet-style trunk network for coordinate processing
2. 3D convolutions with residual connections
3. Superposition of multiple plane waves (n=20) as per theory
4. Proper normalization by local maxima as described
5. Complete inference with MRE calculation and visualization
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
    L = 19.2e-3  # Domain length (m)
    dx = dy = dz = L / Nx
    
    # Wave parameters as per theory
    NUM_WAVES = 20  # n=20 superimposed waves
    THETA_RANGE = (0, np.pi)  # 0 to π
    PHI_RANGE = (-np.pi, np.pi)  # -π to +π
    K_RANGE = (0, 1048)  # k between 0 and 1048
    AMPLITUDE_RANGE = (0, 5)  # Ex, Ey amplitudes
    
    # Dataset parameters
    DATASET_SIZE = 1000
    TRAIN_TEST_SPLIT = 0.8
    BATCH_SIZE = 32
    
    # Training parameters
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.1  # 10% validation from training
    
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
    INFERENCE_K_RANGE = (0.021, 838.34)  # Resulting in 1 MHz to 40 GHz

config = Config()

# ============================================================================
# COORDINATE GRID GENERATION
# ============================================================================

class CoordinateGrid:
    """Manages coordinate grid generation and normalization"""
    
    def __init__(self, nx: int = config.Nx, ny: int = config.Ny, 
                 nz: int = config.Nz, length: float = config.L):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.length = length
        
        # Generate physical coordinates
        self.x = np.linspace(0, length, nx)
        self.y = np.linspace(0, length, ny)
        self.z = np.linspace(0, length, nz)
        
        # Meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Normalized coordinates (0 to 1)
        self.X_norm = self.X / length
        self.Y_norm = self.Y / length
        self.Z_norm = self.Z / length
        
        # Coordinate deltas (Δx, Δy, Δz) for trunk input
        self.dx = self.X_norm.copy()
        self.dy = self.Y_norm.copy()
        self.dz = self.Z_norm.copy()
        
    def get_physical_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return physical coordinates"""
        return self.X, self.Y, self.Z
    
    def get_coordinate_deltas(self) -> np.ndarray:
        """Return coordinate deltas as [3, Nx, Ny, Nz] array (trunk input)"""
        return np.stack([self.dx, self.dy, self.dz], axis=0).astype(np.float32)
    
    def get_normalized_coordinates(self) -> np.ndarray:
        """Return normalized coordinates as [3, Nx, Ny, Nz] array"""
        return np.stack([self.X_norm, self.Y_norm, self.Z_norm], axis=0).astype(np.float32)
    
    def get_grid_spacing(self) -> Tuple[float, float, float]:
        """Return grid spacing"""
        return config.dx, config.dy, config.dz

# Global coordinate grid
grid = CoordinateGrid()

# ============================================================================
# PHYSICS: PLANE WAVE GENERATION AND CURL COMPUTATION
# ============================================================================

class PlaneWaveGenerator:
    """Generates plane wave superpositions as per paper description"""
    
    @staticmethod
    def generate_single_wave(theta: float, phi: float, k: float, 
                            wave_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single plane wave as per equation:
        E_i = E0,i exp[-j k_i (x cosφ sinθ + y sinφ sinθ + z cosθ)]
        
        Returns:
            Ex, Ey, Ez components for wave i
        """
        # Wave vector components
        kx = k * np.sin(theta) * np.cos(phi)
        ky = k * np.sin(theta) * np.sin(phi)
        kz = k * np.cos(theta)
        
        # Random amplitudes for Ex and Ey
        Ex0 = np.random.uniform(*config.AMPLITUDE_RANGE)
        Ey0 = np.random.uniform(*config.AMPLITUDE_RANGE)
        
        # Calculate Ez to satisfy ∇·E = 0 (ki·E0,i = 0)
        if np.abs(np.cos(theta)) > 1e-10:  # Avoid division by zero
            Ez0 = -(np.cos(phi) * np.sin(theta) * Ex0 + 
                    np.sin(phi) * np.sin(theta) * Ey0) / np.cos(theta)
        else:
            Ez0 = 0.0
        
        # Phase term: k·r = kx*x + ky*y + kz*z
        phase = kx * grid.X + ky * grid.Y + kz * grid.Z
        
        # Complex exponential (using Euler's formula for real part)
        Ex = Ex0 * np.cos(phase)
        Ey = Ey0 * np.cos(phase)
        Ez = Ez0 * np.cos(phase)
        
        return Ex, Ey, Ez
    
    @staticmethod
    def generate_wave_superposition(num_waves: int = config.NUM_WAVES) -> Tuple[np.ndarray, Dict]:
        """
        Generate superposition of multiple plane waves as per paper:
        E = Σ_{i=0}^{n} E0,i exp[-j k_i (x cosφ_i sinθ_i + y sinφ_i sinθ_i + z cosθ_i)]
        
        Returns:
            E_total: [3, Nx, Ny, Nz] array of total E-field
            wave_params: Dictionary of wave parameters for reproducibility
        """
        E_total = np.zeros((3, config.Nx, config.Ny, config.Nz), dtype=np.float32)
        wave_params = {
            'thetas': [],
            'phis': [],
            'ks': [],
            'amplitudes': []
        }
        
        for i in range(num_waves):
            # Random parameters as per paper
            theta = np.random.uniform(*config.THETA_RANGE)
            phi = np.random.uniform(*config.PHI_RANGE)
            k = np.random.uniform(*config.K_RANGE)
            
            wave_params['thetas'].append(theta)
            wave_params['phis'].append(phi)
            wave_params['ks'].append(k)
            
            # Generate single wave
            Ex_i, Ey_i, Ez_i = PlaneWaveGenerator.generate_single_wave(theta, phi, k, i)
            
            # Superposition
            E_total[0] += Ex_i
            E_total[1] += Ey_i
            E_total[2] += Ez_i
            
            wave_params['amplitudes'].append((Ex_i.max(), Ey_i.max(), Ez_i.max()))
        
        return E_total.astype(np.float32), wave_params

class CurlCalculator:
    """Computes curl using finite differences with proper boundary handling"""
    
    @staticmethod
    def compute_curl(E: np.ndarray) -> np.ndarray:
        """
        Compute curl using central differences with one-sided differences at boundaries.
        
        Paper uses: ∇×E computed analytically or with high-order FD
        We implement second-order central differences for compatibility.
        
        Args:
            E: [3, Nx, Ny, Nz] E-field components
            
        Returns:
            curlE: [3, Nx, Ny, Nz] curl components
        """
        if E.ndim == 4:
            E = E[np.newaxis, ...]
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, _, Nx, Ny, Nz = E.shape
        curl = np.zeros_like(E)
        
        dx, dy, dz = grid.get_grid_spacing()
        
        # Extract components
        Ex, Ey, Ez = E[:, 0], E[:, 1], E[:, 2]
        
        # ---------- x-derivatives ----------
        dEy_dx = np.zeros_like(Ey)
        dEy_dx[:, 1:-1, :, :] = (Ey[:, 2:, :, :] - Ey[:, :-2, :, :]) / (2 * dx)
        dEy_dx[:, 0, :, :] = (Ey[:, 1, :, :] - Ey[:, 0, :, :]) / dx
        dEy_dx[:, -1, :, :] = (Ey[:, -1, :, :] - Ey[:, -2, :, :]) / dx
        
        dEz_dx = np.zeros_like(Ez)
        dEz_dx[:, 1:-1, :, :] = (Ez[:, 2:, :, :] - Ez[:, :-2, :, :]) / (2 * dx)
        dEz_dx[:, 0, :, :] = (Ez[:, 1, :, :] - Ez[:, 0, :, :]) / dx
        dEz_dx[:, -1, :, :] = (Ez[:, -1, :, :] - Ez[:, -2, :, :]) / dx
        
        # ---------- y-derivatives ----------
        dEx_dy = np.zeros_like(Ex)
        dEx_dy[:, :, 1:-1, :] = (Ex[:, :, 2:, :] - Ex[:, :, :-2, :]) / (2 * dy)
        dEx_dy[:, :, 0, :] = (Ex[:, :, 1, :] - Ex[:, :, 0, :]) / dy
        dEx_dy[:, :, -1, :] = (Ex[:, :, -1, :] - Ex[:, :, -2, :]) / dy
        
        dEz_dy = np.zeros_like(Ez)
        dEz_dy[:, :, 1:-1, :] = (Ez[:, :, 2:, :] - Ez[:, :, :-2, :]) / (2 * dy)
        dEz_dy[:, :, 0, :] = (Ez[:, :, 1, :] - Ez[:, :, 0, :]) / dy
        dEz_dy[:, :, -1, :] = (Ez[:, :, -1, :] - Ez[:, :, -2, :]) / dy
        
        # ---------- z-derivatives ----------
        dEx_dz = np.zeros_like(Ex)
        dEx_dz[:, :, :, 1:-1] = (Ex[:, :, :, 2:] - Ex[:, :, :, :-2]) / (2 * dz)
        dEx_dz[:, :, :, 0] = (Ex[:, :, :, 1] - Ex[:, :, :, 0]) / dz
        dEx_dz[:, :, :, -1] = (Ex[:, :, :, -1] - Ex[:, :, :, -2]) / dz
        
        dEy_dz = np.zeros_like(Ey)
        dEy_dz[:, :, :, 1:-1] = (Ey[:, :, :, 2:] - Ey[:, :, :, :-2]) / (2 * dz)
        dEy_dz[:, :, :, 0] = (Ey[:, :, :, 1] - Ey[:, :, :, 0]) / dz
        dEy_dz[:, :, :, -1] = (Ey[:, :, :, -1] - Ey[:, :, :, -2]) / dz
        
        # ---------- Compute curl ----------
        # curl_x = ∂Ez/∂y - ∂Ey/∂z
        curl[:, 0] = dEz_dy - dEy_dz
        
        # curl_y = ∂Ex/∂z - ∂Ez/∂x
        curl[:, 1] = dEx_dz - dEz_dx
        
        # curl_z = ∂Ey/∂x - ∂Ex/∂y
        curl[:, 2] = dEy_dx - dEx_dy
        
        if squeeze_output:
            curl = curl[0]
        
        return curl.astype(np.float32)

class Normalizer:
    """Handles normalization as per paper: normalize by local maximum"""
    
    @staticmethod
    def normalize_by_local_maximum(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize each component by its local maximum as per paper.
        
        Paper states: "The U-net output, ∇×E, was normalized by the local
        maximum for each component to improve prediction accuracy."
        
        Args:
            field: [3, Nx, Ny, Nz] field to normalize
            
        Returns:
            normalized_field: Normalized field
            max_values: Maximum values for each component
        """
        max_values = np.zeros(3, dtype=np.float32)
        normalized = field.copy()
        
        for c in range(3):
            max_val = np.max(np.abs(field[c]))
            if max_val > 1e-10:
                normalized[c] = field[c] / max_val
                max_values[c] = max_val
            else:
                max_values[c] = 1.0
        
        return normalized.astype(np.float32), max_values
    
    @staticmethod
    def denormalize(normalized_field: np.ndarray, max_values: np.ndarray) -> np.ndarray:
        """Denormalize field using saved maximum values"""
        denormalized = normalized_field.copy()
        for c in range(3):
            denormalized[c] = normalized_field[c] * max_values[c]
        return denormalized

# ============================================================================
# DATASET CLASS
# ============================================================================

class DCODataset(Dataset):
    """Dataset for E-field to curl(E) mapping with coordinate channels"""
    
    def __init__(self, num_samples: int = config.DATASET_SIZE, 
                 mode: str = 'train', norm_stats: Optional[Dict] = None):
        """
        Args:
            num_samples: Number of samples to generate
            mode: 'train', 'val', or 'test'
            norm_stats: Pre-computed normalization statistics
        """
        self.mode = mode
        self.num_samples = num_samples
        
        # Generate coordinate deltas (trunk input) and E-field (branch input)
        self.coords = grid.get_coordinate_deltas()  # [3, Nx, Ny, Nz]
        
        # Generate samples
        self.samples = []
        self.wave_params = []
        
        print(f"Generating {num_samples} {mode} samples...")
        for i in range(num_samples):
            E_field, params = PlaneWaveGenerator.generate_wave_superposition()
            curlE = CurlCalculator.compute_curl(E_field)
            
            # Normalize curl by local maximum as per paper
            curlE_norm, max_vals = Normalizer.normalize_by_local_maximum(curlE)
            
            # Store sample
            self.samples.append({
                'E_field': E_field.astype(np.float32),          # Branch input
                'coords': self.coords.copy(),                    # Trunk input
                'curl_norm': curlE_norm.astype(np.float32),     # Normalized target
                'max_vals': max_vals.astype(np.float32),        # Denormalization factors
                'curl_gt': curlE.astype(np.float32)             # Ground truth
            })
            self.wave_params.append(params)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        # Compute dataset statistics if not provided
        self.norm_stats = norm_stats
        if self.norm_stats is None:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute dataset statistics for reporting"""
        all_curl = np.array([s['curl_gt'] for s in self.samples])
        
        self.norm_stats = {
            'curl_mean': np.mean(all_curl, axis=(0, 2, 3, 4)),
            'curl_std': np.std(all_curl, axis=(0, 2, 3, 4)),
            'curl_max': np.max(np.abs(all_curl), axis=(0, 2, 3, 4)),
            'E_mean': np.mean([s['E_field'] for s in self.samples], axis=(0, 2, 3, 4)),
            'E_std': np.std([s['E_field'] for s in self.samples], axis=(0, 2, 3, 4))
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Return separate inputs for branch and trunk networks
        return (
            torch.from_numpy(sample['E_field']),      # Branch input: [3, 32, 32, 32]
            torch.from_numpy(sample['coords']),       # Trunk input: [3, 32, 32, 32]
            torch.from_numpy(sample['curl_norm']),    # Target: [3, 32, 32, 32]
            torch.from_numpy(sample['max_vals']),     # Max values: [3]
            torch.from_numpy(sample['curl_gt'])       # Ground truth: [3, 32, 32, 32]
        )
    
    def get_statistics(self):
        return self.norm_stats

# ============================================================================
# MODEL ARCHITECTURE: CLEARLY SEPARATED BRANCH, TRUNK, DECODER
# ============================================================================

class ResidualConvBlock3D(nn.Module):
    """
    Standard 3D residual convolutional block used throughout the network.
    
    Exactly matches paper specification:
    1. Conv3D (3×3×3, stride=1, padding=1)
    2. GELU activation
    3. Conv3D (3×3×3, stride=1, padding=1)
    4. GELU activation
    5. Residual connection: output = block(x) + x
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Main convolution path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               padding=1, stride=1)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               padding=1, stride=1)
        self.gelu2 = nn.GELU()
        
        # Shortcut connection if channel dimensions change
        self.shortcut = nn.Identity() if in_channels == out_channels else \
                       nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        # Main path
        out = self.gelu1(self.conv1(x))
        out = self.gelu2(self.conv2(out))
        
        # Residual connection
        return out + residual

class Encoder(nn.Module):
    """
    Shared encoder architecture for both branch (E-field) and trunk (coordinates).
    Produces multi-scale features for skip connections.
    """
    def __init__(self, in_channels: int = 3, init_channels: int = 32, 
                 num_levels: int = 4):
        super().__init__()
        
        self.num_levels = num_levels
        self.channels = [init_channels * (2**i) for i in range(num_levels)]
        
        # Initial convolution
        self.init_conv = ResidualConvBlock3D(in_channels, self.channels[0])
        
        # Downsampling blocks
        self.levels = nn.ModuleList()
        for i in range(num_levels - 1):
            level = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                ResidualConvBlock3D(self.channels[i], self.channels[i+1])
            )
            self.levels.append(level)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            skips: List of skip connections at each level
            bottleneck: Features at lowest resolution
        """
        skips = []
        
        # Initial convolution
        x = self.init_conv(x)
        skips.append(x)  # First skip connection
        
        # Downsampling levels
        for i, level in enumerate(self.levels):
            x = level(x)
            if i < len(self.levels) - 1:  # Don't add bottleneck to skips
                skips.append(x)
        
        return skips, x  # skips: [level0, level1, level2], bottleneck: level3

class DCO_UNet3D(nn.Module):
    """
    DCO-UNet 3D: Differentiable Curl Operator Network

    Architecture:
    - Branch Encoder (E-field)
    - Trunk Encoder (coordinates)
    - Hadamard product at bottleneck ONLY
    - Decoder upsamples first, then concatenates branch skips
    - Residual blocks throughout
    """

    def __init__(
        self,
        branch_in_channels: int = 3,
        trunk_in_channels: int = 3,
        out_channels: int = 3,
        init_channels: int = 32,
        num_levels: int = 4,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.init_channels = init_channels

        # --------------------------------------------------
        # 1. Encoders
        # --------------------------------------------------
        self.branch_encoder = Encoder(
            branch_in_channels, init_channels, num_levels
        )
        self.trunk_encoder = Encoder(
            trunk_in_channels, init_channels, num_levels
        )

        # --------------------------------------------------
        # 2. Bottleneck (branch only)
        # --------------------------------------------------
        bottleneck_channels = init_channels * (2 ** (num_levels - 1))
        self.bottleneck = ResidualConvBlock3D(
            bottleneck_channels,
            bottleneck_channels,
        )

        # --------------------------------------------------
        # 3. Decoder: upsample → concat → residual block
        # --------------------------------------------------
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in reversed(range(num_levels - 1)):
            in_ch = init_channels * (2 ** (i + 1))
            out_ch = init_channels * (2 ** i)

            # Upsampling
            self.upconvs.append(
                nn.ConvTranspose3d(
                    in_ch, out_ch, kernel_size=2, stride=2
                )
            )

            # Decode after concatenation with skip
            self.dec_blocks.append(
                ResidualConvBlock3D(
                    out_ch + out_ch,  # upsampled + skip
                    out_ch,
                )
            )

        # --------------------------------------------------
        # 4. Final projection
        # --------------------------------------------------
        self.final_conv = nn.Conv3d(
            init_channels, out_channels, kernel_size=1
        )

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            branch_input: [B, 3, 32, 32, 32]  (E-field)
            trunk_input:  [B, 3, 32, 32, 32]  (coordinates)

        Returns:
            curl: [B, 3, 32, 32, 32]
        """

        # --------------------------------------------------
        # 1. Encode
        # --------------------------------------------------
        branch_skips, branch_bottleneck = self.branch_encoder(branch_input)
        trunk_skips, trunk_bottleneck = self.trunk_encoder(trunk_input)

        # --------------------------------------------------
        # 2. Bottleneck processing (branch)
        # --------------------------------------------------
        x = self.bottleneck(branch_bottleneck)

        # --------------------------------------------------
        # 3. Hadamard Product (DeepONet core)
        # --------------------------------------------------
        assert x.shape == trunk_bottleneck.shape, (
            f"Hadamard mismatch: "
            f"{x.shape} vs {trunk_bottleneck.shape}"
        )
        x = x * trunk_bottleneck

        # --------------------------------------------------
        # 4. Decoder
        # --------------------------------------------------
        for i in range(len(self.upconvs)):
            # Upsample first
            x = self.upconvs[i](x)

            # Corresponding branch skip
            skip = branch_skips[-1 - i]

            # Safety check (can remove after debugging)
            assert x.shape[2:] == skip.shape[2:], (
                f"Spatial mismatch: "
                f"x={x.shape}, skip={skip.shape}"
            )

            # Concatenate & decode
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)

        # --------------------------------------------------
        # 5. Final output
        # --------------------------------------------------
        return self.final_conv(x)

# ============================================================================
# LOSS FUNCTIONS AND METRICS
# ============================================================================

class CombinedLoss(nn.Module):
    """Combined loss function with MSE and sign-sensitive component"""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        
        # Sign-sensitive loss: encourage same sign patterns
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)
        sign_loss = 1.0 - cosine_sim.mean()
        
        return mse_loss + self.alpha * sign_loss

class Metrics:
    """Metrics calculation including MRE as per paper"""
    
    @staticmethod
    def calculate_mre(pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate Mean Relative Error (MRE) as per paper formula:
        
        MRE = (1/Nx Ny Nz) Σ_{i,j,k=1}^{Nx,Ny,Nz} 
              (y_{i,j,k}^p - y_{i,j,k}^t) / y_{i,j,k}^t  if y_{i,j,k}^t ≠ 0
              y_{i,j,k}^p if y_{i,j,k}^t = 0
              
        where y^t = ground truth, y^p = prediction
        """
        Nx, Ny, Nz = pred.shape[1:]
        total_cells = Nx * Ny * Nz
        
        mre_sum = 0.0
        for c in range(3):  # For each component
            pred_c = pred[c].flatten()
            target_c = target[c].flatten()
            
            for i in range(len(pred_c)):
                if np.abs(target_c[i]) > 1e-10:
                    mre_sum += np.abs(pred_c[i] - target_c[i]) / np.abs(target_c[i])
                else:
                    mre_sum += np.abs(pred_c[i])
        
        return mre_sum / (3 * total_cells)
    
    @staticmethod
    def calculate_component_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate MSE for each component"""
        mse = np.zeros(3)
        for c in range(3):
            mse[c] = np.mean((pred[c] - target[c])**2)
        return mse

# ============================================================================
# TRAINING LOOP
# ============================================================================

class Trainer:
    """Handles model training and validation"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = CombinedLoss(alpha=0.1)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': []
        }
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack batch data
            branch_input, trunk_input, targets = batch_data[0], batch_data[1], batch_data[2]
            branch_input = branch_input.to(self.device)
            trunk_input = trunk_input.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(branch_input, trunk_input)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                branch_input, trunk_input, targets = batch_data[0], batch_data[1], batch_data[2]
                branch_input = branch_input.to(self.device)
                trunk_input = trunk_input.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(branch_input, trunk_input)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = config.EPOCHS) -> Dict:
        """Main training loop"""
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
        
        model_path = config.MODEL_DIR / filename
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")

# ============================================================================
# INFERENCE AND VISUALIZATION
# ============================================================================

class InferenceEngine:
    """Handles model inference and visualization"""
    
    def __init__(self, model_path: str = 'models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DCO_UNet3D().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def generate_test_case(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Generate test case as per paper:
        - delta x = delta y = delta z = 0.6 mm
        - θ = 45°, φ = 60°
        - k: 20 values uniformly selected from 0.021 to 838.34
          (frequency spectrum from 1 MHz to 40 GHz)
        """
        print("\nGenerating inference test case as per paper...")
        
        # Update grid spacing for this test case
        global grid
        L_test = config.INFERENCE_DELTA * config.Nx
        grid = CoordinateGrid(config.Nx, config.Ny, config.Nz, L_test)
        
        # Generate superposition of 20 waves with specified parameters
        k_values = np.linspace(*config.INFERENCE_K_RANGE, config.INFERENCE_K_VALUES)
        
        E_total = np.zeros((3, config.Nx, config.Ny, config.Nz), dtype=np.float32)
        
        for i, k in enumerate(k_values):
            Ex_i, Ey_i, Ez_i = PlaneWaveGenerator.generate_single_wave(
                config.INFERENCE_THETA, config.INFERENCE_PHI, k, i
            )
            E_total[0] += Ex_i
            E_total[1] += Ey_i
            E_total[2] += Ez_i
        
        # Compute analytical curl
        curl_analytical = CurlCalculator.compute_curl(E_total)
        
        # Prepare inputs
        branch_input = E_total  # E-field
        trunk_input = grid.get_coordinate_deltas()  # Coordinates
        
        test_case_info = {
            'delta': config.INFERENCE_DELTA,
            'theta': config.INFERENCE_THETA,
            'phi': config.INFERENCE_PHI,
            'k_values': k_values,
            'frequency_range': (k_values.min() * config.c0 / (2 * np.pi),
                               k_values.max() * config.c0 / (2 * np.pi))
        }
        
        return branch_input, trunk_input, curl_analytical, test_case_info
    
    def predict(self, branch_input: np.ndarray, trunk_input: np.ndarray) -> np.ndarray:
        """Run model inference"""
        # Convert to tensors
        branch_tensor = torch.from_numpy(branch_input).unsqueeze(0).to(self.device)
        trunk_tensor = torch.from_numpy(trunk_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(branch_tensor, trunk_tensor)
        
        # Model outputs normalized curl
        pred_curl_norm = output.squeeze().cpu().numpy()
        
        # Apply post-normalization as per paper
        pred_curl = pred_curl_norm.copy()
        for c in range(3):
            max_val = np.max(np.abs(pred_curl_norm[c]))
            if max_val > 1e-10:
                pred_curl[c] = pred_curl_norm[c] / max_val
        
        return pred_curl
    
    def create_visualizations(self, E_field: np.ndarray, 
                            curl_analytical: np.ndarray, 
                            curl_predicted: np.ndarray,
                            test_info: Dict):
        """Create comprehensive visualizations as per paper requirements"""
        
        # Create figures directory
        config.FIGURES_DIR.mkdir(exist_ok=True)
        
        # 1. Input E-field visualization (x, y, z components)
        self._plot_e_field(E_field, test_info)
        
        # 2. Curl comparison (analytical vs predicted)
        self._plot_curl_comparison(curl_analytical, curl_predicted, test_info)
        
        # 3. MRE heatmap
        self._plot_mre_heatmap(curl_analytical, curl_predicted, test_info)
        
        # 4. Difference heatmap
        self._plot_difference_heatmap(curl_analytical, curl_predicted, test_info)
        
        # 5. Slice visualizations
        self._plot_slices(curl_analytical, curl_predicted, test_info)
    
    def _plot_e_field(self, E_field: np.ndarray, test_info: Dict):
        """Plot input E-field components"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        components = ['Ex', 'Ey', 'Ez']
        slice_idx = config.Nx // 2
        
        vmax = np.max(np.abs(E_field))
        
        for i, (ax, comp) in enumerate(zip(axes, components)):
            im = ax.imshow(E_field[i, :, :, slice_idx], cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
            ax.set_title(f'{comp} (z-slice)')
            ax.set_xlabel('X index')
            ax.set_ylabel('Y index')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)
        
        plt.suptitle(f'Input E-field: θ={np.rad2deg(test_info["theta"]):.1f}°, '
                    f'φ={np.rad2deg(test_info["phi"]):.1f}°, '
                    f'Δ={test_info["delta"]*1000:.1f} mm')
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'input_e_field.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_curl_comparison(self, curl_analytical: np.ndarray, 
                            curl_predicted: np.ndarray, test_info: Dict):
        """Plot analytical vs predicted curl"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        components = ['Curl_x', 'Curl_y', 'Curl_z']
        slice_idx = config.Nx // 2
        
        # Find global vmax for consistent color scaling
        vmax = max(np.max(np.abs(curl_analytical)), np.max(np.abs(curl_predicted)))
        
        for i in range(3):  # Components
            # Analytical
            im1 = axes[i, 0].imshow(curl_analytical[i, :, :, slice_idx], 
                                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, 0].set_title(f'Analytical {components[i]}')
            axes[i, 0].set_xlabel('X index')
            axes[i, 0].set_ylabel('Y index')
            divider = make_axes_locatable(axes[i, 0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im1, cax=cax)
            
            # Predicted
            im2 = axes[i, 1].imshow(curl_predicted[i, :, :, slice_idx], 
                                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[i, 1].set_title(f'Predicted {components[i]}')
            axes[i, 1].set_xlabel('X index')
            axes[i, 1].set_ylabel('Y index')
            divider = make_axes_locatable(axes[i, 1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im2, cax=cax)
            
            # Difference
            diff = curl_predicted[i, :, :, slice_idx] - curl_analytical[i, :, :, slice_idx]
            im3 = axes[i, 2].imshow(diff, cmap='RdBu_r')
            axes[i, 2].set_title(f'Difference {components[i]}')
            axes[i, 2].set_xlabel('X index')
            axes[i, 2].set_ylabel('Y index')
            divider = make_axes_locatable(axes[i, 2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im3, cax=cax)
        
        plt.suptitle('Curl Comparison: Analytical vs Predicted (Middle z-slice)')
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'curl_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mre_heatmap(self, curl_analytical: np.ndarray, 
                         curl_predicted: np.ndarray, test_info: Dict):
        """Plot MRE heatmap as per paper formula"""
        # Calculate MRE for each spatial point
        Nx, Ny, Nz = curl_analytical.shape[1:]
        mre_map = np.zeros((Nx, Ny, Nz))
        
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    mre_sum = 0
                    for c in range(3):
                        target = curl_analytical[c, i, j, k]
                        pred = curl_predicted[c, i, j, k]
                        
                        if np.abs(target) > 1e-10:
                            mre_sum += np.abs(pred - target) / np.abs(target)
                        else:
                            mre_sum += np.abs(pred)
                    
                    mre_map[i, j, k] = mre_sum / 3
        
        # Calculate overall MRE
        overall_mre = Metrics.calculate_mre(curl_predicted, curl_analytical)
        
        # Plot slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slice_idx = config.Nx // 2
        
        # XY slice
        im1 = axes[0].imshow(mre_map[:, :, slice_idx], cmap='hot_r')
        axes[0].set_title('MRE Heatmap (XY slice)')
        axes[0].set_xlabel('X index')
        axes[0].set_ylabel('Y index')
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # XZ slice
        im2 = axes[1].imshow(mre_map[:, slice_idx, :], cmap='hot_r')
        axes[1].set_title('MRE Heatmap (XZ slice)')
        axes[1].set_xlabel('X index')
        axes[1].set_ylabel('Z index')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im2, cax=cax)
        
        # YZ slice
        im3 = axes[2].imshow(mre_map[slice_idx, :, :], cmap='hot_r')
        axes[2].set_title('MRE Heatmap (YZ slice)')
        axes[2].set_xlabel('Y index')
        axes[2].set_ylabel('Z index')
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im3, cax=cax)
        
        plt.suptitle(f'Mean Relative Error Distribution\nOverall MRE: {overall_mre:.4f}')
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'mre_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return overall_mre
    
    def _plot_difference_heatmap(self, curl_analytical: np.ndarray, 
                               curl_predicted: np.ndarray, test_info: Dict):
        """Plot absolute difference heatmap"""
        # Calculate absolute difference
        diff = np.abs(curl_predicted - curl_analytical)
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=0))  # Vector magnitude
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slice_idx = config.Nx // 2
        
        # XY slice
        im1 = axes[0].imshow(diff_magnitude[:, :, slice_idx], cmap='viridis')
        axes[0].set_title('Difference Magnitude (XY slice)')
        axes[0].set_xlabel('X index')
        axes[0].set_ylabel('Y index')
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # XZ slice
        im2 = axes[1].imshow(diff_magnitude[:, slice_idx, :], cmap='viridis')
        axes[1].set_title('Difference Magnitude (XZ slice)')
        axes[1].set_xlabel('X index')
        axes[1].set_ylabel('Z index')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im2, cax=cax)
        
        # YZ slice
        im3 = axes[2].imshow(diff_magnitude[slice_idx, :, :], cmap='viridis')
        axes[2].set_title('Difference Magnitude (YZ slice)')
        axes[2].set_xlabel('Y index')
        axes[2].set_ylabel('Z index')
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im3, cax=cax)
        
        plt.suptitle('Absolute Difference Heatmap: |Predicted - Analytical|')
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'difference_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_slices(self, curl_analytical: np.ndarray, 
                    curl_predicted: np.ndarray, test_info: Dict):
        """Plot multiple slices for detailed comparison"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        slice_positions = [8, 16, 24]
        
        vmax = max(np.max(np.abs(curl_analytical)), np.max(np.abs(curl_predicted)))
        
        for comp_idx in range(3):  # Components
            for slice_idx, pos in enumerate(slice_positions):
                # Analytical
                im1 = axes[comp_idx, slice_idx].imshow(
                    curl_analytical[comp_idx, :, :, pos], 
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax
                )
                axes[comp_idx, slice_idx].set_title(
                    f'Curl_{"xyz"[comp_idx]} (z={pos}), Analytical'
                )
                axes[comp_idx, slice_idx].set_xlabel('X index')
                axes[comp_idx, slice_idx].set_ylabel('Y index')
                divider = make_axes_locatable(axes[comp_idx, slice_idx])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im1, cax=cax)
            
            # Predicted at middle slice
            im2 = axes[comp_idx, 3].imshow(
                curl_predicted[comp_idx, :, :, config.Nx//2], 
                cmap='RdBu_r', vmin=-vmax, vmax=vmax
            )
            axes[comp_idx, 3].set_title(f'Curl_{"xyz"[comp_idx]}, Predicted (z={config.Nx//2})')
            axes[comp_idx, 3].set_xlabel('X index')
            axes[comp_idx, 3].set_ylabel('Y index')
            divider = make_axes_locatable(axes[comp_idx, 3])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im2, cax=cax)
        
        plt.suptitle('Detailed Slice Comparison at Different Z-positions')
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / 'detailed_slices.png', dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("DCO-UNet 3D: Differentiable Curl Operator Learning")
    print("Based on: 'A Differentiable Curl Operator for Deep Learning Electromagnetics'")
    print("="*70)
    
    # Create output directories
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    
    # 1. Generate dataset
    print("\n1. Generating dataset...")
    full_dataset = DCODataset(num_samples=config.DATASET_SIZE, mode='full')
    
    # Split dataset
    train_size = int(config.DATASET_SIZE * config.TRAIN_TEST_SPLIT)
    val_size = int(config.DATASET_SIZE * config.VAL_SPLIT)
    test_size = config.DATASET_SIZE - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # 2. Initialize model and trainer
    print("\n2. Initializing model...")
    model = DCO_UNet3D(
        branch_in_channels=3,
        trunk_in_channels=3,
        out_channels=3,
        init_channels=config.INIT_CHANNELS,
        num_levels=config.NUM_LEVELS
    )
    
    # Print model summary
    print(f"\nModel Architecture Summary:")
    print(f"  Branch input channels: 3 (Ex, Ey, Ez)")
    print(f"  Trunk input channels: 3 (Δx, Δy, Δz)")
    print(f"  Output channels: 3 (curl_x, curl_y, curl_z)")
    print(f"  Initial channels: {config.INIT_CHANNELS}")
    print(f"  Number of levels: {config.NUM_LEVELS}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(model, config)
    
    # 3. Train model
    print("\n3. Training model...")
    history = trainer.train(train_loader, val_loader, epochs=config.EPOCHS)
    
    # 4. Plot training history
    print("\n4. Plotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    # Calculate and plot MSE
    if 'train_mse' in history:
        axes[1].plot(history['train_mse'], label='Train MSE')
        axes[1].plot(history['val_mse'], label='Validation MSE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('MSE History')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / 'training_history.png', dpi=300)
    plt.close()
    
    # 5. Run inference on test case
    print("\n5. Running inference on test case...")
    inference_engine = InferenceEngine('models/best_model.pth')
    
    # Generate test case
    branch_input, trunk_input, curl_analytical, test_info = inference_engine.generate_test_case()
    
    # Run prediction
    curl_predicted = inference_engine.predict(branch_input, trunk_input)
    
    # Calculate metrics
    mre = Metrics.calculate_mre(curl_predicted, curl_analytical)
    mse_components = Metrics.calculate_component_mse(curl_predicted, curl_analytical)
    
    print(f"\nTest Case Results:")
    print(f"  Mean Relative Error (MRE): {mre:.6f}")
    print(f"  Component MSE:")
    print(f"    X: {mse_components[0]:.6e}")
    print(f"    Y: {mse_components[1]:.6e}")
    print(f"    Z: {mse_components[2]:.6e}")
    
    print(f"\nTest Case Parameters:")
    print(f"  Δx = Δy = Δz = {test_info['delta']*1000:.1f} mm")
    print(f"  θ = {np.rad2deg(test_info['theta']):.1f}°")
    print(f"  φ = {np.rad2deg(test_info['phi']):.1f}°")
    print(f"  Frequency range: {test_info['frequency_range'][0]/1e6:.1f} MHz "
          f"to {test_info['frequency_range'][1]/1e9:.1f} GHz")
    print(f"  k range: {test_info['k_values'].min():.3f} to {test_info['k_values'].max():.3f} rad/m")
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    inference_engine.create_visualizations(branch_input, curl_analytical, 
                                          curl_predicted, test_info)
    
    # 7. Save results
    print("\n7. Saving results...")
    results = {
        'test_info': {
            'delta_mm': float(test_info['delta'] * 1000),
            'theta_deg': float(np.rad2deg(test_info['theta'])),
            'phi_deg': float(np.rad2deg(test_info['phi'])),
            'k_min': float(test_info['k_values'].min()),
            'k_max': float(test_info['k_values'].max()),
            'freq_min_MHz': float(test_info['frequency_range'][0] / 1e6),
            'freq_max_GHz': float(test_info['frequency_range'][1] / 1e9)
        },
        'metrics': {
            'mre': float(mre),
            'mse_components': mse_components.tolist()
        },
        'training_history': {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
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
