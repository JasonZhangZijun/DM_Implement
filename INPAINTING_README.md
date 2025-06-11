# 🎨 DDPM图像修复(Inpainting)使用指南

这是一个基于DDPM（Denoising Diffusion Probabilistic Models）的图像修复实现，可以智能地填补图像中缺失的区域。

## 📋 功能特点

- 🖼️ **多种遮罩类型**: 支持中心方形、左/上半边、随机区域、条纹等遮罩（由 `utils/masks.py` 统一实现）
- 🔄 **改进算法**: 提供标准和改进两种inpainting算法
- 🎲 **多样本生成**: 可为同一遮罩图像生成多个不同的修复结果
- 📊 **可视化对比**: 自动生成原图、遮罩图、修复图的对比结果
- 🧮 **质量评估**: 内置 `evaluate_quality`，支持 **PSNR / SSIM**（见 `utils/losses.py`）
- 🚀 **易于使用**: 简单的API接口，支持命令行和编程调用

## 🚀 快速开始

### 1. 基本使用

```bash
# 运行基本演示
python main.py

# 使用自定义参数
python main.py --image_path your_image.jpg --use_improved
```

### 2. 完整功能演示

```bash
# 运行所有演示功能
python inpainting_usage.py
```

## 💻 编程接口

### 初始化模型

```python
from inpaint_ddpm import InpaintDDPM

# 创建数据加载器（必需）
class DummyDataLoader:
    def __init__(self):
        self.batch_size = 4
    def __iter__(self):
        for i in range(2):
            yield torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))

# 初始化模型
dataloader = DummyDataLoader()
model = InpaintDDPM(dataloader, T=50, device='cuda')

# 加载预训练模型（可选）
model.load_pretrained_model('ddpm_model.pth')
```

### 基本使用方法

```python
# 1. 从图像文件直接修复
model.inpaint_from_image(
    'path/to/image.jpg',
    mask_type='center',
    mask_size=0.5,
    save_path='result.png'
)

# 2. 手动创建遮罩和修复
image = model.load_and_preprocess_image('image.jpg')
mask = model.create_mask(image.shape, 'center', 0.4)
result = model.inpaint_improved(image, mask)

# 3. 生成多个修复结果
results = model.inpaint_improved(image, mask, num_samples=3)

# 4. 计算质量指标
metrics = model.evaluate_quality(image, results[0:1])
print(metrics)  # {'PSNR': xx.xx, 'SSIM': xx.xx}
```

## 🎭 遮罩类型

| 类型 | 描述 | 示例用途 |
|------|------|----------|
| `center` | 中心方形区域 | 移除图像中心的物体 |
| `left_half` | 左半边区域 | 图像左侧修复 |
| `top_half` | 上半边区域 | 图像上方修复 |
| `random` | 随机散布区域 | 修复噪点或随机损坏 |
| `stripes` | 条纹遮罩 | 修复扫描线或条纹损坏 |

## 🔧 算法选择

### 标准算法 (`inpaint`)
- 基础的DDPM inpainting实现
- 在每步约束已知区域到噪声版本的原图

### 改进算法 (`inpaint_improved`)  
- 更稳定的约束策略
- 通常产生更好的修复效果
- **推荐使用**

## 📁 项目结构

```
diffusion/
├── inpaint_ddpm.py          # 主要的inpainting实现
├── main.py                  # 简单的命令行接口
├── inpainting_usage.py      # 完整功能演示
├── models/
│   └── ddpm.py             # 基础DDPM模型
├── utils/
│   ├── masks.py          # 遮罩工具
│   └── losses.py         # PSNR & SSIM 指标
└── 生成的图像文件/
    ├── inpaint_demo_*.png   # 基本演示结果
    ├── custom_mask_*.png    # 自定义遮罩结果
    ├── multiple_samples_*.png # 多样本结果
    └── algorithm_comparison_*.png # 算法对比结果
```

## 🎯 使用示例

### 示例1: 简单修复

```python
# 最简单的使用方式
model.inpaint_from_image('photo.jpg')
```

### 示例2: 自定义遮罩

```python
# 创建大的中心遮罩
model.inpaint_from_image(
    'photo.jpg',
    mask_type='center',
    mask_size=0.7,
    save_path='big_center_inpaint.png'
)
```

### 示例3: 多样本生成

```python
# 为同一图像生成3个不同的修复结果
image = model.load_and_preprocess_image('photo.jpg')
mask = model.create_mask(image.shape, 'random', 0.3)
results = model.inpaint_improved(image, mask, num_samples=3)

# 保存每个结果
for i, result in enumerate(results):
    model.save_inpaint_results(
        image, image*mask, result[i:i+1], mask,
        f'result_{i+1}.png'
    )
```

### 示例4: 算法对比

```python
# 比较两种算法的效果
image = model.load_and_preprocess_image('photo.jpg')
mask = model.create_mask(image.shape, 'center', 0.5)

result_std = model.inpaint(image, mask)
result_imp = model.inpaint_improved(image, mask)

# 保存对比结果
model.save_inpaint_results(image, image*mask, result_std, mask, 'standard.png')
model.save_inpaint_results(image, image*mask, result_imp, mask, 'improved.png')
```

## ⚙️ 参数说明

### InpaintDDPM 初始化参数
- `dataloader`: 数据加载器（必需）
- `T`: 扩散步数，默认1000（建议测试时使用50）
- `beta_start`: 噪声调度起始值，默认0.0001
- `beta_end`: 噪声调度结束值，默认0.02
- `device`: 计算设备，默认自动选择

### 修复方法参数
- `mask_type`: 遮罩类型
- `mask_size`: 遮罩大小比例 (0-1)
- `num_samples`: 生成样本数量
- `use_improved`: 是否使用改进算法

## 🚨 注意事项

1. **模型要求**: 当前实现支持32x32像素图像，其他尺寸会被自动调整
2. **计算资源**: 建议使用GPU加速，CPU运行会较慢
3. **预训练模型**: 为获得最佳效果，建议使用在相关数据集上预训练的DDPM模型
4. **时间步数**: 更大的T值产生更好质量但需要更长时间

## 🔧 扩展和自定义

### 添加新的遮罩类型

```python
def create_custom_mask(self, image_shape):
    # 在InpaintDDPM类中添加新的遮罩逻辑
    mask = torch.ones(image_shape)
    # 自定义遮罩逻辑
    return mask
```

### 调整算法参数

```python
# 修改扩散参数
model = InpaintDDPM(
    dataloader, 
    T=100,                    # 减少步数加快速度
    beta_start=0.0001,       # 调整噪声调度
    beta_end=0.02
)
```

## 📊 结果分析与指标

### 1. 视觉对比
生成的对比图包含四个部分：
1. **原始图像**: 未被遮罩的完整图像
2. **遮罩图像**: 应用遮罩后的图像（黑色区域为待修复）
3. **修复结果**: AI 生成的修复图像
4. **遮罩可视化**: 遮罩的可视化（白色=保留，黑色=修复）

### 2. 数值指标
调用 `model.evaluate_quality(original, reconstructed)` 可一次性获得 PSNR 与 SSIM，用于量化对比不同算法或参数配置：

```python
metrics = model.evaluate_quality(original, reconstructed)
print(f"PSNR: {metrics['PSNR']:.2f} dB, SSIM: {metrics['SSIM']:.4f}")
```

## 🤝 贡献

欢迎提交问题报告、功能请求或改进建议！

## 🎯 许可证

本项目仅供学习和研究使用。 