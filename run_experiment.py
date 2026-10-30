import os
import math
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class ColorizationDataset(Dataset):
    def __init__(self, sketch_paths, color_paths, transform=None):
        self.sketch_paths = sketch_paths
        self.color_paths = color_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.sketch_paths)
        
    def __getitem__(self, idx):
        sketch = Image.open(self.sketch_paths[idx]).convert('RGB')
        color = Image.open(self.color_paths[idx]).convert('RGB')
        
        if self.transform:
            sketch = self.transform(sketch)
            color = self.transform(color)
            
        return sketch, color, os.path.basename(self.sketch_paths[idx])

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

sketch_dir = 'DataSet/sketch'
color_dir = 'DataSet/color'
valid_exts = ('.png', '.jpg', '.jpeg')
filenames = sorted([f for f in os.listdir(sketch_dir) if f.lower().endswith(valid_exts)])

all_sketches = [os.path.join(sketch_dir, f) for f in filenames]
all_colors = [os.path.join(color_dir, f) for f in filenames]

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

# 1. Save dataset_samples.png
sample_dataset = ColorizationDataset(all_sketches, all_colors, transform=transform)
fig, axes = plt.subplots(3, 4, figsize=(12, 8.5))
sample_indices = [0, 11, 20, 31, 40, 51]  # Four & March samples across Apple, Carrot, Flower

for i, idx in enumerate(sample_indices):
    s_tensor, c_tensor, name = sample_dataset[idx]
    row = i // 2
    col_s = (i % 2) * 2
    col_c = col_s + 1
    
    axes[row, col_s].imshow(denormalize(s_tensor))
    axes[row, col_s].set_title(f'Sketch: {name}', fontsize=11, fontweight='bold')
    axes[row, col_s].axis('off')
    
    axes[row, col_c].imshow(denormalize(c_tensor))
    axes[row, col_c].set_title(f'Target: {name}', fontsize=11, fontweight='bold', color='#1d4ed8')
    axes[row, col_c].axis('off')

plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=250, bbox_inches='tight')
plt.close()

# 2. Train / Test Split (80/20)
indices = np.arange(len(all_sketches))
np.random.shuffle(indices)

split_point = int(0.8 * len(indices))
train_indices = indices[:split_point]
test_indices = indices[split_point:]

train_sketches = [all_sketches[i] for i in train_indices]
train_colors = [all_colors[i] for i in train_indices]
test_sketches = [all_sketches[i] for i in test_indices]
test_colors = [all_colors[i] for i in test_indices]

train_dataset = ColorizationDataset(train_sketches, train_colors, transform=transform)
test_dataset = ColorizationDataset(test_sketches, test_colors, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 3. Model
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBlock(3, 32)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)
        
        self.b = ConvBlock(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d1 = ConvBlock(64, 32)
        
        self.out = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.p1(x1))
        x3 = self.e3(self.p2(x2))
        
        x_mid = self.b(self.p3(x3))
        
        y3 = self.up3(x_mid)
        y3 = self.d3(torch.cat([y3, x3], dim=1))
        y2 = self.up2(y3)
        y2 = self.d2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.d1(torch.cat([y1, x1], dim=1))
        
        return self.out(y1)

model = UNetColorizer().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 35
train_losses = []
test_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    running_train_loss = 0.0
    for sketches, colors, _ in train_loader:
        sketches = sketches.to(device)
        colors = colors.to(device)
        
        optimizer.zero_grad()
        outputs = model(sketches)
        loss = criterion(outputs, colors)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * sketches.size(0)
    
    train_loss = running_train_loss / len(train_dataset)
    train_losses.append(train_loss)
    
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for sketches, colors, _ in test_loader:
            sketches = sketches.to(device)
            colors = colors.to(device)
            outputs = model(sketches)
            loss = criterion(outputs, colors)
            running_test_loss += loss.item() * sketches.size(0)
            
    test_loss = running_test_loss / len(test_dataset)
    test_losses.append(test_loss)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

torch.save(model.state_dict(), 'colorizer_unet.pth')

# 4. Save loss_curve.png
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, epochs + 1), train_losses, label='Train L1 Loss', color='#2563eb', linewidth=2.5)
plt.plot(range(1, epochs + 1), test_losses, label='Test L1 Loss', color='#ea580c', linewidth=2.5)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('L1 Loss (MAE)', fontsize=12)
plt.title('Training & Test Loss Curve (U-Net Colorizer)', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=250, bbox_inches='tight')
plt.close()

# 5. Evaluate Metrics
def calculate_psnr(target, pred):
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

model.eval()
all_mae = []
all_psnr = []

with torch.no_grad():
    for sketches, colors, names in test_loader:
        sketches = sketches.to(device)
        colors = colors.to(device)
        outputs = model(sketches)
        
        for i in range(sketches.size(0)):
            pred_img = denormalize(outputs[i])
            target_img = denormalize(colors[i])
            
            mae = np.mean(np.abs(target_img - pred_img))
            psnr = calculate_psnr(target_img, pred_img)
            
            all_mae.append(float(mae))
            all_psnr.append(float(psnr))

avg_mae = float(np.mean(all_mae))
avg_psnr = float(np.mean(all_psnr))
print(f'Average MAE:  {avg_mae:.4f}')
print(f'Average PSNR: {avg_psnr:.2f} dB')

# 6. Save colorization_results.png (6 test samples) & slide_results_grid.png (horizontal 4 samples for slide)
num_show = min(6, len(test_dataset))
fig, axes = plt.subplots(num_show, 3, figsize=(10, 3.0 * num_show))

with torch.no_grad():
    for i in range(num_show):
        s_tensor, c_tensor, name = test_dataset[i]
        input_batch = s_tensor.unsqueeze(0).to(device)
        pred_batch = model(input_batch)
        
        s_img = denormalize(s_tensor)
        c_img = denormalize(c_tensor)
        p_img = denormalize(pred_batch[0])
        
        ax_row = axes[i]
        ax_row[0].imshow(s_img)
        ax_row[0].set_title(f'Input Sketch ({name})', fontsize=11)
        ax_row[0].axis('off')
        
        ax_row[1].imshow(p_img)
        ax_row[1].set_title('Model Output (Colorized)', fontsize=11, color='#15803d', fontweight='bold')
        ax_row[1].axis('off')
        
        ax_row[2].imshow(c_img)
        ax_row[2].set_title('Ground Truth Target', fontsize=11, color='#1d4ed8')
        ax_row[2].axis('off')

plt.tight_layout()
plt.savefig('colorization_results.png', dpi=250, bbox_inches='tight')
plt.close()

# Also create a compact 3x4 grid for the 16:9 slide (4 test cases, rows = Sketch, Predicted, Target)
fig, axes = plt.subplots(3, 4, figsize=(12, 7.5))
row_labels = ['Input Sketch', 'Model Output', 'Ground Truth']
with torch.no_grad():
    for col in range(4):
        s_tensor, c_tensor, name = test_dataset[col]
        pred_batch = model(s_tensor.unsqueeze(0).to(device))
        imgs = [denormalize(s_tensor), denormalize(pred_batch[0]), denormalize(c_tensor)]
        for row in range(3):
            axes[row, col].imshow(imgs[row])
            if row == 0:
                axes[row, col].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('slide_results_grid.png', dpi=250, bbox_inches='tight')
plt.close()

with open('metrics_summary.json', 'w') as f:
    json.dump({
        'avg_mae': avg_mae,
        'avg_psnr': avg_psnr,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }, f, indent=2)
print('Done all exports!')

