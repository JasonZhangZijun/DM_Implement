# Diffusion Models Project Guide

This project includes implementations of various diffusion models, including DDPM, DDIM, conditional diffusion models, and image inpainting models. Below are the functionality descriptions and running methods for all major Python files.

## 📁 Project Structure

```
diffusion/
├── models/                     # Model implementations
│   ├── ddpm.py                # DDPM core implementation
│   ├── ddim.py                # DDIM implementation
│   ├── unet.py                # U-Net architecture
│   ├── improved_unet.py       # Improved U-Net
│   ├── conditional_ddpm.py    # Conditional DDPM
│   ├── conditional_unet.py    # Conditional U-Net
│   └── super_resolution.py    # Super-resolution model
├── utils/                      # Utility functions
│   ├── fid.py                 # FID evaluation
│   ├── losses.py              # Loss functions
│   └── masks.py               # Mask generation
├── config.py                  # Configuration file
└── Main running scripts...
```

## 🚀 Main Executable Files

### 1. Unified Training and Evaluation Script ⭐ **Recommended**

**File:** `train_and_evaluate.py`  
**Function:** Complete training, evaluation, and sampling pipeline, supporting multiple datasets and configurations

```bash
# Train CIFAR-10 model
python train_and_evaluate.py --task cifar10

# Train CelebA model
python train_and_evaluate.py --task celeba

# Use custom configuration
python train_and_evaluate.py --task cifar10 --config custom_config.yaml
```

**Output:**
- Training logs and loss curves
- Periodically saved model checkpoints
- Generated sample images
- FID evaluation results

---

### 2. Basic DDPM Training Scripts

**File:** `train_ddpm_cifar.py`  
**Function:** Train basic DDPM model on CIFAR-10 dataset

```bash
python train_ddpm_cifar.py
```

**File:** `train_ddpm_celeba.py`  
**Function:** Train basic DDPM model on CelebA dataset

```bash
python train_ddpm_celeba.py
```

**Output:**
- `diffusion_model_cifar.pth` or `diffusion_model_celeba.pth`
- `sample.png` (generated samples)
- FID evaluation scores

---

### 3. DDIM Model Examples

**File:** `ddim_example.py`  
**Function:** Train and test DDIM model, supports fast sampling

```bash
python ddim_example.py
```

**File:** `train_ddim_cifar.py` ⭐ **New**  
**Function:** Specialized DDIM model training on CIFAR-10 dataset

```bash
python train_ddim_cifar.py
```

**File:** `train_ddim_celeba.py` ⭐ **New**  
**Function:** Specialized DDIM model training on CelebA dataset (supports 64x64 images)

```bash
python train_ddim_celeba.py
```

**Output:**
- `ddim_model.pth` / `ddim_model_cifar.pth` / `ddim_model_celeba.pth` (model weights)
- `ddim_sample_standard.png` (standard 50-step sampling)
- `ddim_sample_fast.png` (fast 10-step sampling)  
- `ddim_sample_stochastic.png` (stochastic sampling)
- `ddim_*_batch.png` (large batch samples, CelebA only)

---

### 4. Conditional Diffusion Models

**File:** `conditional_diffusion.py`  
**Function:** Class-label based conditional generation model

```bash
# Train conditional model
python conditional_diffusion.py --train --epochs 20

# Generate specific class images (e.g., generate 10 car images, class_id=1)
python conditional_diffusion.py --sample --num_samples 10 --class_id 1

# Train and sample simultaneously
python conditional_diffusion.py --train --sample --epochs 10 --class_id 3
```

**CIFAR-10 Class Labels:**
- 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer
- 5: dog, 6: frog, 7: horse, 8: ship, 9: truck

**Output:**
- `conditional_diffusion.pth` (model weights)
- `cond_samples/class_{class_id}.png` (conditional generated samples)

---

### 5. Image Inpainting

**File:** `main.py`  
**Function:** Simple image inpainting demonstration

```bash
# Basic inpainting demo
python main.py

# Specify pretrained model
python main.py --model_path diffusion_model_cifar.pth

# Specify input image
python main.py --image_path your_image.jpg --output_dir results
```

**File:** `inpaint_cifar10.py`  
**Function:** Image inpainting on CIFAR-10 dataset

```bash
python inpaint_cifar10.py
```

**File:** `inpaint_celeba.py`  
**Function:** Face inpainting on CelebA dataset

```bash
python inpaint_celeba.py
```

**File:** `inpainting_usage.py`  
**Function:** Complete image inpainting tool, supporting various mask types

```bash
# View usage instructions
python inpainting_usage.py --help

# Basic usage
python inpainting_usage.py --input_dir ./images --output_dir ./results
```

**Output:**
- Before and after inpainting comparison images
- Inpainting results with different mask types

---

### 6. DDIM Testing Scripts

**File:** `test_ddim.py`  
**Function:** Test different sampling strategies for DDIM model

```bash
python test_ddim.py
```

**File:** `test_ddpm_cifar.py` ⭐ **New**  
**Function:** Test pretrained DDPM CIFAR-10 model

```bash
python test_ddpm_cifar.py
```

**File:** `compare_ddim_ddpm.py` ⭐ **New**  
**Function:** Complete comparison of DDIM and DDPM on both datasets

```bash
python compare_ddim_ddpm.py
```

**File:** `quick_comparison.py` ⭐ **New Recommended**  
**Function:** Quick comparison test, runnable even without pretrained models

```bash
python quick_comparison.py
```

**Output:**
- Sampling results comparison with different steps
- Sampling time statistics
- `ddim_vs_ddpm_comparison.png` (comprehensive comparison image)
- `ddim_ddpm_quick_comparison.png` (quick comparison image)
- `ddpm_cifar_test_samples.png` (DDPM CIFAR-10 test samples)
- Detailed performance analysis report

---

## ⚙️ Configuration File

**File:** `config.py`  
**Function:** Contains default configuration parameters for all models and training

Main configuration items:
- Model architecture parameters (UNet channels, attention layers, etc.)
- Training parameters (learning rate, batch size, number of epochs, etc.)
- Diffusion process parameters (number of timesteps T, noise schedule, etc.)
- Dataset configuration (image size, data path, etc.)

---

## 🔧 Model Files Description

This section details the core modules in the `models/` and `utils/` directories, which are the foundation for all training and inference tasks.

### Core Models (`models/` directory)

- **`ddpm.py`**:
  - **Function**: Implements standard DDPM (Denoising Diffusion Probabilistic Models).
  - **Core**: Contains complete logic for `q_sample` (forward noising process) and `p_sample_loop` (reverse denoising sampling). This is the foundation for all diffusion models.

- **`ddim.py`**:
  - **Function**: Implements DDIM (Denoising Diffusion Implicit Models), a faster sampling method.
  - **Core**: Provides deterministic sampling process, allowing high-quality image generation in fewer steps through `ddim_step`.

- **`unet.py`**:
  - **Function**: Provides basic U-Net network architecture for noise prediction in diffusion process.
  - **Architecture**: Uses classic encoder-decoder structure with skip connections to preserve multi-scale features. Fixed upsampling layer and skip connection dimension matching issues.

- **`improved_unet.py`**:
  - **Function**: Implements improved U-Net architecture.
  - **Improvements**: Introduces ResBlock and AttentionBlock, enhancing feature extraction capability and global information perception, typically achieving better generation results.

- **`conditional_ddpm.py`**:
  - **Function**: Inherits from `DDPM_Model`, implements class-conditional diffusion model.
  - **Core**: During training and sampling, receives class label `y` as additional input besides timestep `t` to generate class-specific images.

- **`conditional_unet.py`**:
  - **Function**: U-Net for use with `ConditionalDDPM_Model`.
  - **Architecture**: Adds label embedding module to standard U-Net. Converts class labels to vectors and fuses with time embedding, injecting into network's deepest layer to guide generation process.

- **`super_resolution.py`**:
  - **Function**: Implements diffusion model for image super-resolution tasks.
  - **Core**: Uses low-resolution image as condition, combined with noise input, to generate corresponding high-resolution image.

### Utility Files (`utils/` directory)

- **`fid.py`**:
  - **Function**: Calculates FID (Fréchet Inception Distance) score, a common metric for evaluating generated image quality and diversity.
  - **Implementation**: Contains simplified InceptionV3 feature extraction network for extracting features from real and generated images and calculating distribution distance. Lower scores indicate higher image quality.

- **`losses.py`**:
  - **Function**: Provides common image quality evaluation loss functions.
  - **Core Metrics**:
    - `psnr`: Calculates Peak Signal-to-Noise Ratio, measuring image reconstruction quality.
    - `ssim`: Calculates Structural Similarity Index, measuring image similarity in terms of brightness, contrast, and structure.
  - **Features**: All implementations are based on PyTorch with no external dependencies.

- **`masks.py`**:
  - **Function**: Mask generation and application tools designed for image inpainting tasks.
  - **Core Functions**:
    - `create_mask`: A unified entry point for generating various mask types, such as `center_mask`, `random_mask`, `left_half_mask`, etc.
    - `apply_mask`: Applies generated mask to image.
  - **Features**: `1` represents known regions, `0` represents unknown regions to be inpainted.

---

## 📊 Model Weight Files

The project includes the following pretrained models:
- `diffusion_model.pth`: General DDPM model
- `diffusion_model_cifar.pth`: CIFAR-10 specific model
- `diffusion_model_celeba.pth`: CelebA specific model
- `ddim_model.pth`: DDIM model weights

---

## 🎯 Quick Start

1. **Simplest Usage** (Recommended for beginners):
   ```bash
   python train_and_evaluate.py --task cifar10
   ```

2. **Conditional Generation** (Generate specific class images):
   ```bash
   python conditional_diffusion.py --train --sample --class_id 2
   ```

3. **Image Inpainting**:
   ```bash
   python main.py --model_path diffusion_model_cifar.pth
   ```

4. **Fast Sampling** (DDIM):
   ```bash
   python ddim_example.py
   ```

5. **Model Comparison** ⭐ **New Recommended**:
   ```bash
   python quick_comparison.py
   ```

---

## 📋 Dependencies

Ensure the following Python packages are installed:
```bash
pip install torch torchvision
pip install numpy matplotlib
pip install scipy pillow
pip install kagglehub  # for dataset download
```

---

## 🎨 Output Files Description

- **`sample.png`**: Generated sample image grid
- **`experiments/`**: Training experiment results directory
- **`*_outputs/`**: Various output image directories
- **`*.pth`**: PyTorch model weight files
- **Log files**: Contains training losses and evaluation metrics

---

## 💡 Usage Tips

1. **First Time Use**: Recommended to start with `train_and_evaluate.py`
2. **Quick Test**: Use `ddim_example.py` for fast sampling experiments
3. **Specific Tasks**: Choose corresponding specialized scripts based on needs
4. **Custom Configuration**: Modify parameters in `config.py` to suit your needs

For any questions, please check other documentation files in the project:
- `PROJECT_SUMMARY.md`: Project summary
- `INPAINTING_README.md`: Detailed image inpainting instructions
- `todo.md`: Development plans and known issues 