# 扩散模型项目使用指南

这个项目包含了多种扩散模型的实现，包括 DDPM、DDIM、条件扩散模型和图像修复模型。以下是所有主要 Python 文件的功能说明和运行方法。

## 📁 项目结构

```
diffusion/
├── models/                     # 模型实现
│   ├── ddpm.py                # DDPM 核心实现
│   ├── ddim.py                # DDIM 实现
│   ├── unet.py                # U-Net 网络架构
│   ├── improved_unet.py       # 改进的 U-Net
│   ├── conditional_ddpm.py    # 条件 DDPM
│   ├── conditional_unet.py    # 条件 U-Net
│   └── super_resolution.py   # 超分辨率模型
├── utils/                      # 工具函数
│   ├── fid.py                 # FID 评估
│   ├── losses.py              # 损失函数
│   └── masks.py               # 掩码生成
├── config.py                  # 配置文件
└── 主要运行脚本...
```

## 🚀 主要可运行文件

### 1. 统一训练和评估脚本 ⭐ **推荐使用**

**文件：** `train_and_evaluate.py`  
**功能：** 完整的训练、评估和采样流程，支持多种数据集和配置

```bash
# 训练 CIFAR-10 模型
python train_and_evaluate.py --task cifar10

# 训练 CelebA 模型
python train_and_evaluate.py --task celeba

# 使用自定义配置
python train_and_evaluate.py --task cifar10 --config custom_config.yaml
```

**输出：**
- 训练日志和损失曲线
- 定期保存的模型检查点
- 生成样本图像
- FID 评估结果

---

### 2. 基础 DDPM 训练脚本

**文件：** `train_ddpm_cifar.py`  
**功能：** 在 CIFAR-10 数据集上训练基础 DDPM 模型

```bash
python train_ddpm_cifar.py
```

**文件：** `train_ddpm_celeba.py`  
**功能：** 在 CelebA 数据集上训练基础 DDPM 模型

```bash
python train_ddpm_celeba.py
```

**输出：**
- `diffusion_model_cifar.pth` 或 `diffusion_model_celeba.pth`
- `sample.png` (生成样本)
- FID 评估分数

---

### 3. DDIM 模型示例

**文件：** `ddim_example.py`  
**功能：** 训练和测试 DDIM 模型，支持快速采样

```bash
python ddim_example.py
```

**文件：** `train_ddim_cifar.py` ⭐ **新增**  
**功能：** 专门在 CIFAR-10 数据集上训练 DDIM 模型

```bash
python train_ddim_cifar.py
```

**文件：** `train_ddim_celeba.py` ⭐ **新增**  
**功能：** 专门在 CelebA 数据集上训练 DDIM 模型（支持64x64图像）

```bash
python train_ddim_celeba.py
```

**输出：**
- `ddim_model.pth` / `ddim_model_cifar.pth` / `ddim_model_celeba.pth` (模型权重)
- `ddim_sample_standard.png` (标准50步采样)
- `ddim_sample_fast.png` (快速10步采样)  
- `ddim_sample_stochastic.png` (随机性采样)
- `ddim_*_batch.png` (大批量样本，仅CelebA)

---

### 4. 条件扩散模型

**文件：** `conditional_diffusion.py`  
**功能：** 基于类别标签的条件生成模型

```bash
# 训练条件模型
python conditional_diffusion.py --train --epochs 20

# 生成特定类别的图像（如生成10张汽车图像，class_id=1）
python conditional_diffusion.py --sample --num_samples 10 --class_id 1

# 同时训练和采样
python conditional_diffusion.py --train --sample --epochs 10 --class_id 3
```

**CIFAR-10 类别标签：**
- 0: 飞机, 1: 汽车, 2: 鸟, 3: 猫, 4: 鹿
- 5: 狗, 6: 青蛙, 7: 马, 8: 船, 9: 卡车

**输出：**
- `conditional_diffusion.pth` (模型权重)
- `cond_samples/class_{class_id}.png` (条件生成样本)

---

### 5. 图像修复 (Inpainting)

**文件：** `main.py`  
**功能：** 简单的图像修复演示

```bash
# 基础修复演示
python main.py

# 指定预训练模型
python main.py --model_path diffusion_model_cifar.pth

# 指定输入图像
python main.py --image_path your_image.jpg --output_dir results
```

**文件：** `inpaint_cifar10.py`  
**功能：** CIFAR-10 数据集上的图像修复

```bash
python inpaint_cifar10.py
```

**文件：** `inpaint_celeba.py`  
**功能：** CelebA 数据集上的人脸修复

```bash
python inpaint_celeba.py
```

**文件：** `inpainting_usage.py`  
**功能：** 完整的图像修复工具，支持多种掩码类型

```bash
# 查看使用说明
python inpainting_usage.py --help

# 基础使用
python inpainting_usage.py --input_dir ./images --output_dir ./results
```

**输出：**
- 修复前后对比图像
- 不同掩码类型的修复结果

---

### 6. DDIM 测试脚本

**文件：** `test_ddim.py`  
**功能：** 测试 DDIM 模型的不同采样策略

```bash
python test_ddim.py
```

**文件：** `test_ddpm_cifar.py` ⭐ **新增**  
**功能：** 测试预训练的 DDPM CIFAR-10 模型

```bash
python test_ddpm_cifar.py
```

**文件：** `compare_ddim_ddpm.py` ⭐ **新增**  
**功能：** 完整对比 DDIM 和 DDPM 在两个数据集上的效果

```bash
python compare_ddim_ddpm.py
```

**文件：** `quick_comparison.py` ⭐ **新增推荐**  
**功能：** 快速对比测试，即使没有预训练模型也能运行

```bash
python quick_comparison.py
```

**输出：**
- 不同步数的采样结果对比
- 采样时间统计
- `ddim_vs_ddpm_comparison.png` (综合对比图像)
- `ddim_ddpm_quick_comparison.png` (快速对比图像)
- `ddpm_cifar_test_samples.png` (DDPM CIFAR-10 测试样本)
- 详细的性能分析报告

---

## ⚙️ 配置文件

**文件：** `config.py`  
**功能：** 包含所有模型和训练的默认配置参数

主要配置项：
- 模型架构参数 (UNet 通道数、注意力层等)
- 训练参数 (学习率、批次大小、epoch 数等)
- 扩散过程参数 (时间步数 T、噪声调度等)
- 数据集配置 (图像大小、数据路径等)

---

## 🔧 模型文件说明

这部分详细介绍了 `models/` 和 `utils/` 目录下的核心模块，它们是构建所有训练和推理任务的基础。

### 核心模型 (`models/` 目录)

- **`ddpm.py`**:
  - **功能**: 实现了标准的 DDPM (Denoising Diffusion Probabilistic Models)。
  - **核心**: 包含 `q_sample` (前向加噪过程) 和 `p_sample_loop` (反向去噪采样) 的完整逻辑。这是所有扩散模型的基础。

- **`ddim.py`**:
  - **功能**: 实现了 DDIM (Denoising Diffusion Implicit Models)，它是一种更快的采样方法。
  - **核心**: 提供了确定性采样过程，允许在更少的步骤内生成高质量图像，通过 `ddim_step` 实现。

- **`unet.py`**:
  - **功能**: 提供了基础的 U-Net 网络架构，用于在扩散过程中预测噪声。
  - **架构**: 采用经典的编码器-解码器结构，并通过跳跃连接（skip connections）保留多尺度特征。已修复了上采样层与跳跃连接的尺寸匹配问题。

- **`improved_unet.py`**:
  - **功能**: 实现了改进版的 U-Net 架构。
  - **改进点**: 引入了残差块 (ResBlock) 和自注意力机制 (AttentionBlock)，增强了模型的特征提取能力和对全局信息的感知，通常能获得更好的生成效果。

- **`conditional_ddpm.py`**:
  - **功能**: 继承自 `DDPM_Model`，实现了类别条件扩散模型。
  - **核心**: 在训练和采样时，除了时间步 `t`，还会接收类别标签 `y` 作为额外输入，以生成特定类别的图像。

- **`conditional_unet.py`**:
  - **功能**: 配合 `ConditionalDDPM_Model` 使用的 U-Net。
  - **架构**: 在标准 U-Net 的基础上增加了类别嵌入 (label embedding) 模块。它将类别标签转换为向量，并与时间嵌入融合，注入到网络的最深层，从而引导生成过程。

- **`super_resolution.py`**:
  - **功能**: 实现了用于图像超分辨率任务的扩散模型。
  - **核心**: 将低分辨率图像作为条件，结合噪声输入，生成对应的高分辨率图像。

### 工具文件 (`utils/` 目录)

- **`fid.py`**:
  - **功能**: 用于计算 FID (Fréchet Inception Distance) 分数，这是一个评估生成图像质量和多样性的常用指标。
  - **实现**: 包含一个简化的 InceptionV3 特征提取网络，用于从真实图像和生成图像中提取特征，并计算它们分布之间的距离。分数越低，表示生成图像的质量越高。

- **`losses.py`**:
  - **功能**: 提供了常用的图像质量评估损失函数。
  - **核心指标**:
    - `psnr`: 计算峰值信噪比 (Peak Signal-to-Noise Ratio)，衡量图像重建质量。
    - `ssim`: 计算结构相似性 (Structural Similarity Index)，从亮度、对比度和结构三方面衡量图像相似度。
  - **特点**: 所有实现都基于 PyTorch，没有外部依赖。

- **`masks.py`**:
  - **功能**: 专为图像修复 (Inpainting) 任务设计的掩码（mask）生成和应用工具。
  - **核心函数**:
    - `create_mask`: 一个统一的入口，可以生成多种类型的掩码，如 `center_mask` (中心遮挡), `random_mask` (随机遮挡), `left_half_mask` (左半边遮挡) 等。
    - `apply_mask`: 将生成的掩码应用到图像上。
  - **特点**: `1` 代表已知区域，`0` 代表需要修复的未知区域。

---

## 📊 模型权重文件

项目中包含以下预训练模型：
- `diffusion_model.pth`: 通用 DDPM 模型
- `diffusion_model_cifar.pth`: CIFAR-10 专用模型
- `diffusion_model_celeba.pth`: CelebA 专用模型
- `ddim_model.pth`: DDIM 模型权重

---

## 🎯 快速开始

1. **最简单的使用方式**（推荐新手）：
   ```bash
   python train_and_evaluate.py --task cifar10
   ```

2. **条件生成**（生成特定类别图像）：
   ```bash
   python conditional_diffusion.py --train --sample --class_id 2
   ```

3. **图像修复**：
   ```bash
   python main.py --model_path diffusion_model_cifar.pth
   ```

4. **快速采样**（DDIM）：
   ```bash
   python ddim_example.py
   ```

5. **模型效果对比** ⭐ **新增推荐**：
   ```bash
   python quick_comparison.py
   ```

---

## 📋 依赖要求

确保安装以下 Python 包：
```bash
pip install torch torchvision
pip install numpy matplotlib
pip install scipy pillow
pip install kagglehub  # 用于数据集下载
```

---

## 🎨 输出文件说明

- **`sample.png`**: 生成的样本图像网格
- **`experiments/`**: 训练实验结果目录
- **`*_outputs/`**: 各种输出图像目录
- **`*.pth`**: PyTorch 模型权重文件
- **日志文件**: 包含训练损失和评估指标

---

## 💡 使用建议

1. **首次使用**：建议从 `train_and_evaluate.py` 开始
2. **快速测试**：使用 `ddim_example.py` 进行快速采样实验
3. **特定任务**：根据需求选择对应的专用脚本
4. **自定义配置**：修改 `config.py` 中的参数以适应你的需求

有任何问题，请查看项目中的其他文档文件：
- `PROJECT_SUMMARY.md`: 项目总结
- `INPAINTING_README.md`: 图像修复详细说明
- `todo.md`: 开发计划和已知问题 