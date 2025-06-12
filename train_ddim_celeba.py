# DDIM 模型训练 - CelebA 数据集
from models.unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.ddim import DDIM_Model
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

### 1. 准备 CelebA 数据集
transform = T.Compose([
    T.CenterCrop(178),        # CelebA 的人脸裁剪
    T.Resize((64, 64)),       # 64x64 分辨率
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(root='./data/celeba', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"CelebA 数据集大小: {len(dataset)}")

### 自定义 DDIM 模型以支持 64x64 图像
class DDIM_Model_CelebA(DDIM_Model):
    def __init__(self, dataloader, T=1000, beta_start=0.0001, beta_end=0.02, device=None):
        super().__init__(dataloader, T, beta_start, beta_end, device)
    
    @torch.no_grad()
    def sample(self, num_samples, ddim_steps=50, eta=0.0):
        """生成 64x64 CelebA 样本"""
        shape = (num_samples, 3, 64, 64)  # CelebA 64x64 图像尺寸
        return self.p_sample_loop(shape, ddim_steps, eta)
    
    @torch.no_grad()
    def fast_sample(self, num_samples, ddim_steps=10, eta=0.0):
        """快速采样 64x64 图像"""
        return self.sample(num_samples, ddim_steps, eta)

### 2. 训练 DDIM 模型
if __name__ == "__main__":
    print("开始训练 DDIM 模型 (CelebA)...")
    
    # 创建 CelebA 专用 DDIM 模型
    ddim_model = DDIM_Model_CelebA(dataloader, T=1000, device=device)
    
    # 训练模型 (CelebA 通常需要更少的 epoch 因为数据集更大)
    print("正在训练模型...")
    ddim_model.train_model(num_epochs=5, lr=1e-4, save_path="ddim_model_celeba.pth")
    
    # 保存模型状态字典
    torch.save(ddim_model.state_dict(), "ddim_model_celeba.pth")
    print("模型已保存为: ddim_model_celeba.pth")
    
    # 创建新模型并加载状态字典
    ddim_model_loaded = DDIM_Model_CelebA(dataloader, T=1000, device=device)
    ddim_model_loaded.load_state_dict(torch.load("ddim_model_celeba.pth"))
    
    print("开始生成人脸样本...")
    
    # 使用标准 DDIM 采样生成样本 (50步)
    print("生成标准 DDIM 人脸样本 (50步)...")
    x_standard = ddim_model_loaded.sample(16, ddim_steps=50, eta=0.0)
    torchvision.utils.save_image(x_standard, "ddim_celeba_standard.png", normalize=True, nrow=4)
    
    # 使用快速 DDIM 采样生成样本 (10步)
    print("生成快速 DDIM 人脸样本 (10步)...")
    x_fast = ddim_model_loaded.fast_sample(16, ddim_steps=10, eta=0.0)
    torchvision.utils.save_image(x_fast, "ddim_celeba_fast.png", normalize=True, nrow=4)
    
    # 使用随机性采样 (eta > 0)
    print("生成随机性 DDIM 人脸样本...")
    x_stochastic = ddim_model_loaded.sample(16, ddim_steps=25, eta=0.5)
    torchvision.utils.save_image(x_stochastic, "ddim_celeba_stochastic.png", normalize=True, nrow=4)
    
    # 生成更多样本用于展示
    print("生成大批量人脸样本...")
    x_batch = ddim_model_loaded.sample(64, ddim_steps=30, eta=0.2)
    torchvision.utils.save_image(x_batch, "ddim_celeba_batch.png", normalize=True, nrow=8)
    
    # 评估模型
    print("开始评估 DDIM 模型...")
    fid_score = ddim_model_loaded.evaluate(num_samples=1000)
    print(f"最终 FID 分数: {fid_score:.2f}")
    
    print("\n🎉 DDIM CelebA 训练完成！")
    print("生成的图像文件:")
    print("- ddim_celeba_standard.png (标准50步采样)")
    print("- ddim_celeba_fast.png (快速10步采样)")
    print("- ddim_celeba_stochastic.png (随机性采样)")
    print("- ddim_celeba_batch.png (64张人脸样本)")
    print("模型权重: ddim_model_celeba.pth")