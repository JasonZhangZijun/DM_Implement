import torch
import os

class BaseConfig:
    """基础配置类"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "outputs"
        self.batch_size = 32
        self.num_workers = 4
        self.seed = 42
        
        # DDPM通用参数
        self.T = 1000  # 扩散步数
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # 训练参数
        self.num_epochs = 100
        self.lr = 1e-4
        self.save_interval = 10
        self.eval_interval = 5
        
        # 采样参数
        self.sample_size = 16
        self.ddim_eta = 0.0
        self.ddim_steps = 50

class CIFAR10Config(BaseConfig):
    """CIFAR-10数据集配置"""
    def __init__(self):
        super().__init__()
        self.dataset_name = "cifar10"
        self.image_size = 32
        self.image_channels = 3
        self.num_classes = 10
        self.data_path = "./data/cifar10"
        
        # UNet架构参数
        self.unet_channels = [64, 128, 256, 512]
        self.attention_resolutions = [16, 8]
        
        # 训练特定参数
        self.batch_size = 128
        self.num_epochs = 200

class InpaintConfig(BaseConfig):
    """图像修复任务配置"""
    def __init__(self):
        super().__init__()
        self.task_name = "inpainting"
        
        # 修复特定参数
        self.mask_types = ["center", "left_half", "top_half", "random", "stripes"]
        self.mask_ratios = [0.25, 0.5, 0.75]
        
        # 改进算法参数
        self.use_improved_algorithm = True
        self.constraint_strength = 0.1
        
        # 评估参数
        self.eval_num_samples = 100
        self.calculate_fid = True
        self.calculate_psnr_ssim = True

class ConditionalConfig(BaseConfig):
    """条件生成配置"""
    def __init__(self):
        super().__init__()
        self.task_name = "conditional"
        
        # 条件生成参数
        self.condition_type = "class"  # "class", "text", "image"
        self.num_classes = 10
        self.class_embed_dim = 128
        
        # 无分类器引导参数
        self.classifier_free_guidance = True
        self.guidance_scale = 7.5
        self.unconditional_prob = 0.1

class SuperResolutionConfig(BaseConfig):
    """超分辨率配置"""
    def __init__(self):
        super().__init__()
        self.task_name = "super_resolution"
        
        # 超分参数
        self.low_res_size = 16
        self.high_res_size = 64
        self.scale_factor = 4
        
        # 数据增强
        self.use_random_crop = True
        self.use_flip = True

def get_config(task="cifar10"):
    """根据任务名称获取对应配置"""
    config_map = {
        "cifar10": CIFAR10Config,
        "inpainting": InpaintConfig,
        "conditional": ConditionalConfig,
        "super_resolution": SuperResolutionConfig,
    }
    
    if task not in config_map:
        raise ValueError(f"未知任务: {task}. 支持的任务: {list(config_map.keys())}")
    
    return config_map[task]()

# 使用示例:
# config = get_config("cifar10")
# print(f"使用设备: {config.device}")
# print(f"图像尺寸: {config.image_size}") 