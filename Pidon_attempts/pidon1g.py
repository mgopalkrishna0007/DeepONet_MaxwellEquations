import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------------
# Physical constants & Grid
# -------------------------------
c0 = 3e8
pi = np.pi
Nx = Ny = Nz = 32
L = 19.2e-3
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
z = np.linspace(0, L, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Pre-compute coordinate tensor for concatenation: Shape [3, 32, 32, 32]
# Normalize coordinates to range [0, 1] for better neural net stability
COORD_GRID = np.stack([X, Y, Z], axis=0) / L 
COORD_GRID = COORD_GRID.astype(np.float32)

# -------------------------------
# Plane Wave Generator (From your code)
# -------------------------------
def generate_plane_wave():
    theta = np.random.uniform(0, pi)
    phi   = np.random.uniform(0, 2 * pi)
    f = np.random.uniform(1e6, 50e9)
    k0 = 2 * pi * f / c0
    kx = k0 * np.sin(theta) * np.cos(phi)
    ky = k0 * np.sin(theta) * np.sin(phi)
    kz = k0 * np.cos(theta)
    
    Ex0 = np.random.randn()
    Ey0 = np.random.randn()
    Ez0 = -(kx * Ex0 + ky * Ey0) / kz if abs(kz) > 1e-6 else 0.0
    E0 = np.array([Ex0, Ey0, Ez0])
    
    phase = kx * X + ky * Y + kz * Z
    Ex = E0[0] * np.cos(phase)
    Ey = E0[1] * np.cos(phase)
    Ez = E0[2] * np.cos(phase)
    E = np.stack([Ex, Ey, Ez], axis=0)
    
    curl_x = ky * Ez - kz * Ey
    curl_y = kz * Ex - kx * Ez
    curl_z = kx * Ey - ky * Ex
    curlE = np.stack([curl_x, curl_y, curl_z], axis=0)
    
    return E.astype(np.float32), curlE.astype(np.float32)

class EMFieldDataset(Dataset):
    def __init__(self, num_samples):
        self.data = []
        print(f"Generating {num_samples} samples...")
        for _ in range(num_samples):
            E, curlE = generate_plane_wave()
            
            # Concatenate Inputs: [Ex, Ey, Ez] + [x, y, z] -> [6, 32, 32, 32]
            # Coordinates are repeated for every sample to provide spatial context
            Input_Tensor = np.concatenate([COORD_GRID, E], axis=0)
            
            self.data.append((Input_Tensor, curlE))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.from_numpy(x), torch.from_numpy(y)



class ResidualBlock(nn.Module):
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
        return out + residual  # Residual connection

class DCO_UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Config ---
        # Input: 6 channels (x, y, z, Ex, Ey, Ez)
        # Output: 3 channels (curl_x, curl_y, curl_z)
        features = [32, 64, 128, 256] # Feature map depths for 4 levels
        
        # --- Encoder (Down-sampling) ---
        # Level 1 (Input -> 32)
        self.enc1_conv = nn.Sequential(
            nn.Conv3d(6, features[0], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[0])
        )
        self.pool1 = nn.MaxPool3d(2)

        # Level 2 (32 -> 64)
        self.enc2_conv = nn.Sequential(
            nn.Conv3d(features[0], features[1], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[1])
        )
        self.pool2 = nn.MaxPool3d(2)

        # Level 3 (64 -> 128)
        self.enc3_conv = nn.Sequential(
            nn.Conv3d(features[1], features[2], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[2])
        )
        self.pool3 = nn.MaxPool3d(2)

        # Level 4 (128 -> 256)
        self.enc4_conv = nn.Sequential(
            nn.Conv3d(features[2], features[3], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[3])
        )
        self.pool4 = nn.MaxPool3d(2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features[3], features[3]*2, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[3]*2)
        )

        # --- Decoder (Up-sampling) ---
        # Level 4
        self.up4 = nn.ConvTranspose3d(features[3]*2, features[3], kernel_size=2, stride=2)
        self.dec4_conv = nn.Sequential(
            nn.Conv3d(features[3]*2, features[3], kernel_size=3, padding=1), # *2 due to skip cat
            nn.GELU(),
            ResidualBlock(features[3])
        )

        # Level 3
        self.up3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv3d(features[2]*2, features[2], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[2])
        )

        # Level 2
        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv3d(features[1]*2, features[1], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[1])
        )

        # Level 1
        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv3d(features[0]*2, features[0], kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(features[0])
        )

        # Final Output Layer (Map back to 3 channels)
        self.final_conv = nn.Conv3d(features[0], 3, kernel_size=1)

    def forward(self, x):
        # Encoder
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

        # Decoder with Skip Connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1) # Skip connection
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


def train_model():
    # 1. Setup Data
    full_dataset = EMFieldDataset(num_samples=1000)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCO_UNet3D().to(device)
    
    # 3. Optimization
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("Starting training on:", device)
    
    # Training Loop
    epochs = 20 # Reduced from 1000 for demonstration; set to 1000 for full run
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- Normalization Strategy ---
            # "Normalize predicted curl during training... local component-wise maximum"
            # We normalize the TARGET to match the scale of neural net outputs (usually ~0-1 range)
            # Then we can un-normalize predictions later.
            
            # Find max abs value per sample for normalization
            # Shape: [B, 1, 1, 1, 1]
            max_val = torch.amax(torch.abs(targets), dim=(1,2,3,4), keepdim=True) + 1e-8
            targets_norm = targets / max_val
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss computed on normalized fields
            loss = criterion(outputs, targets_norm)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.6f}")

    return model, test_loader, device


def visualize_results(model, test_loader, device):
    model.eval()
    
    # Get one batch
    inputs, targets = next(iter(test_loader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        
        # Un-normalize if we were strictly enforcing the training norm strategy output
        # But since model learned to map Input -> Normalized Target, 
        # we strictly compare Model Output vs Normalized Target for error visualization
        # OR we multiply output by max_val to get physical units back.
        
        # Here we visualize the raw comparison (Normalized Space) for clarity of model performance
        pass

    # Move to CPU for plotting
    input_field = inputs[0].cpu().numpy() # [6, 32, 32, 32]
    target_curl = targets[0].cpu().numpy() # [3, 32, 32, 32]
    pred_curl = outputs[0].cpu().numpy()   # [3, 32, 32, 32]
    
    # Normalize targets locally for visualization consistency if not already done
    max_t = np.max(np.abs(target_curl))
    target_curl /= max_t
    pred_curl /= max_t # Assuming model learned normalized space

    # Slice Index (Middle of the cube)
    mid = 16
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    # Row 1: Ground Truth Curl (x, y, z components)
    axes[0,0].imshow(target_curl[0, :, :, mid], cmap='jet')
    axes[0,0].set_title('GT Curl X (Slice Z=16)')
    axes[0,1].imshow(target_curl[1, :, :, mid], cmap='jet')
    axes[0,1].set_title('GT Curl Y')
    axes[0,2].imshow(target_curl[2, :, :, mid], cmap='jet')
    axes[0,2].set_title('GT Curl Z')

    # Row 2: Predicted Curl
    axes[1,0].imshow(pred_curl[0, :, :, mid], cmap='jet')
    axes[1,0].set_title('Pred Curl X')
    axes[1,1].imshow(pred_curl[1, :, :, mid], cmap='jet')
    axes[1,1].set_title('Pred Curl Y')
    axes[1,2].imshow(pred_curl[2, :, :, mid], cmap='jet')
    axes[1,2].set_title('Pred Curl Z')

    # Row 3: Absolute Error
    err = np.abs(target_curl - pred_curl)
    axes[2,0].imshow(err[0, :, :, mid], cmap='inferno')
    axes[2,0].set_title('Error X')
    axes[2,1].imshow(err[1, :, :, mid], cmap='inferno')
    axes[2,1].set_title('Error Y')
    axes[2,2].imshow(err[2, :, :, mid], cmap='inferno')
    axes[2,2].set_title('Error Z')

    plt.tight_layout()
    plt.show()

# Run the pipeline
if __name__ == "__main__":
    trained_model, test_loader, dev = train_model()
    visualize_results(trained_model, test_loader, dev)