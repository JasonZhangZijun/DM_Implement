# -*- coding: utf-8 -*-
"""
Test pre-trained DDIM model on CelebA dataset
"""

import torch
import torchvision
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
import numpy as np
from models.ddim import DDIM_Model
from torchvision.datasets import ImageFolder

class DDIM_Model_CelebA(DDIM_Model):
    def __init__(self, dataloader, T=1000, beta_start=0.0001, beta_end=0.02, device=None):
        super().__init__(dataloader, T, beta_start, beta_end, device)
    
    @torch.no_grad()
    def sample(self, num_samples, ddim_steps=50, eta=0.0):
        """Generate 64x64 CelebA samples"""
        shape = (num_samples, 3, 64, 64)  # CelebA 64x64 image size
        return self.p_sample_loop(shape, ddim_steps, eta)

def create_comparison_grid(model_path="ddim_model_celeba.pth", num_samples=4, output_path="ddim_celeba_comparison.png"):
    """
    Create a comparison grid showing sampling quality evolution
    Including initial noise and samples with different steps
    
    Args:
        model_path: Path to pre-trained model
        num_samples: Number of samples per configuration
        output_path: Path for output comparison image
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run train_ddim_celeba.py first")
        return
    
    # Create dataloader with fake dataset for model initialization
    class FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return torch.randn(3, 64, 64), 0  # CelebA size: 64x64
        def __len__(self):
            return num_samples
            
    dataloader = torch.utils.data.DataLoader(FakeDataset(), batch_size=num_samples)

    # Load DDIM model
    print(f"Loading model: {model_path}")
    model = DDIM_Model_CelebA(dataloader, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate initial noise
    torch.manual_seed(42)  # Fix random seed for reproducibility
    initial_noise = torch.randn(num_samples, 3, 64, 64).to(device)
    
    # Sampling configurations
    sampling_configs = [
        {"name": "20 step", "steps": 20, "eta": 0.0},
        {"name": "50 step", "steps": 50, "eta": 0.0},
        {"name": "100 step", "steps": 1000, "eta": 0.0},
        {"name": "50 step(random)", "steps": 50, "eta": 0.5}
    ]
    
    # Store all generated images
    all_images = []
    all_titles = []
    
    # Add initial noise
    all_images.append(initial_noise.cpu())
    all_titles.append("Initial Noise")
    
    print("Generating samples with different steps...")
    
    with torch.no_grad():
        for config in sampling_configs:
            print(f"Generating {config['name']} samples...")
            
            # Use same initial noise
            torch.manual_seed(42)
            
            # Generate samples
            samples = model.sample(num_samples, ddim_steps=config["steps"], eta=config["eta"])
            
            all_images.append(samples.cpu())
            all_titles.append(f"{config['name']}")
    
    # Create comparison grid
    fig, axes = plt.subplots(len(all_images), num_samples, figsize=(num_samples * 3, len(all_images) * 3))
    fig.suptitle('DDIM CelebA Sampling Quality Evolution', fontsize=16, fontweight='bold')
    
    for row_idx, (images, title) in enumerate(zip(all_images, all_titles)):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[row_idx]
            
            # Convert image format for display
            img = images[col_idx]
            img = torch.clamp((img + 1) / 2, 0, 1)  # Convert from [-1,1] to [0,1]
            img = img.permute(1, 2, 0).numpy()
            
            ax.imshow(img)
            ax.axis('off')
            
            # Add row titles on first column
            if col_idx == 0:
                ax.text(-0.1, 0.5, title, transform=ax.transAxes, 
                       rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            
            # Add column titles on first row
            if row_idx == 0:
                ax.set_title(f'Sample {col_idx + 1}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison grid saved: {output_path}")
    print("The grid shows the evolution from initial noise to final samples")

def test_celeba_ddim(model_path="ddim_model_celeba.pth", num_samples=16, output_prefix="ddim_celeba_test"):
    """
    Test DDIM model on CelebA dataset with various sampling configurations
    
    Args:
        model_path: Path to pre-trained model
        num_samples: Number of samples to generate
        output_prefix: Prefix for output files
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please run train_ddim_celeba.py first")
        return
    
    class FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return torch.randn(3, 64, 64), 0  # CelebA size: 64x64
        def __len__(self):
            return num_samples
            
    dataloader = torch.utils.data.DataLoader(FakeDataset(), batch_size=num_samples)

    # Load DDIM model
    print(f"Loading model: {model_path}")
    model = DDIM_Model_CelebA(dataloader, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Generating {num_samples} samples...")

    # Test different DDIM sampling configurations
    sampling_configs = [
        {"name": "fast", "steps": 20, "eta": 0.0, "desc": "fast (20 step)"},
        {"name": "standard", "steps": 50, "eta": 0.0, "desc": "standard (50 step)"},
        {"name": "high_quality", "steps": 1000, "eta": 0.0, "desc": "high quality (100 step)"},
        {"name": "stochastic", "steps": 50, "eta": 0.5, "desc": "stochastic (50 step, eta=0.5)"}
    ]
    
    with torch.no_grad():
        for config in sampling_configs:
            print(f"Generating {config['desc']}...")
            
            # Generate samples
            samples = model.sample(num_samples, ddim_steps=config["steps"], eta=config["eta"])
            
            # Save images
            output_path = f"{output_prefix}_{config['name']}.png"
            torchvision.utils.save_image(samples, output_path, nrow=4, normalize=True, value_range=(-1, 1))
            print(f"✅ Saved: {output_path}")
    
    print("\n🎉 DDIM CelebA test complete!")
    print("Generated files:")
    for config in sampling_configs:
        print(f"  - {output_prefix}_{config['name']}.png ({config['desc']})")

def compare_ddim_speeds(model_path="ddim_model_celeba.pth", num_samples=8):
    """
    Compare generation speed with different DDIM steps
    """
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    class FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            return torch.randn(3, 64, 64), 0  # CelebA size: 64x64
        def __len__(self):
            return num_samples
            
    dataloader = torch.utils.data.DataLoader(FakeDataset(), batch_size=num_samples)
    
    model = DDIM_Model_CelebA(dataloader, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n⏱️  DDIM Speed Comparison:")
    print("=" * 40)
    
    step_configs = [10, 20, 50, 100, 200, 500]
    shape = (num_samples, 3, 64, 64)  # CelebA size: 64x64
    
    with torch.no_grad():
        for steps in step_configs:
            start_time = time.time()
            samples = model.sample(num_samples, ddim_steps=steps, eta=0.0)
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"{steps:3d} steps: {elapsed:.2f}s ({elapsed/num_samples:.3f}s/image)")

if __name__ == "__main__":
    # Create step comparison grid
    print("Creating DDIM step quality comparison grid...")
    create_comparison_grid()
    
    print("\n" + "="*50)
    
    # Basic test
    test_celeba_ddim()
    
    # Speed comparison test
    print("\n" + "="*50)
    compare_ddim_speeds() 