# DDIM 模型训练 - CIFAR-10 数据集
from models.unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.ddim import DDIM_Model
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

### 1. 准备 CIFAR-10 数据集
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

### 2. 训练 DDIM 模型
if __name__ == "__main__":
    print("开始训练 DDIM 模型 (CIFAR-10)...")
    
    # 创建 DDIM 模型
    ddim_model = DDIM_Model(dataloader, T=1000, device=device)
    
    # 训练模型
    print("正在训练模型...")
    ddim_model.train_model(num_epochs=50, lr=1e-4, save_path="ddim_model_cifar.pth")
    
    # 保存模型状态字典
    torch.save(ddim_model.state_dict(), "ddim_model_cifar.pth")
    print("模型已保存为: ddim_model_cifar.pth")
    
    # # 创建新模型并加载状态字典
    ddim_model_loaded = DDIM_Model(dataloader, T=1000, device=device)
    ddim_model_loaded.load_state_dict(torch.load("ddim_model_cifar.pth"))
    
    print("开始生成样本...")
    
    # 使用标准 DDIM 采样生成样本 (50步)
    print("生成标准 DDIM 样本 (50步)...")
    x_standard = ddim_model_loaded.sample(10, ddim_steps=200, eta=0.0)
    torchvision.utils.save_image(x_standard, "ddim_cifar_standard.png", normalize=True, nrow=4)
    
    # 使用快速 DDIM 采样生成样本 (10步)
    print("生成快速 DDIM 样本 (10步)...")
    x_fast = ddim_model_loaded.fast_sample(10, ddim_steps=500, eta=0.0)
    torchvision.utils.save_image(x_fast, "ddim_cifar_fast.png", normalize=True, nrow=4)
    
    # 使用随机性采样 (eta > 0)
    print("生成随机性 DDIM 样本...")
    x_stochastic = ddim_model_loaded.sample(10, ddim_steps=1000, eta=0.5)
    torchvision.utils.save_image(x_stochastic, "ddim_cifar_stochastic.png", normalize=True, nrow=4)
    
    # 评估模型
    print("开始评估 DDIM 模型...")
    fid_score = ddim_model_loaded.evaluate(num_samples=1000)
    print(f"最终 FID 分数: {fid_score:.2f}")
    
    print("\n🎉 DDIM CIFAR-10 训练完成！")
    print("生成的图像文件:")
    print("- ddim_cifar_standard.png (标准50步采样)")
    print("- ddim_cifar_fast.png (快速10步采样)")
    print("- ddim_cifar_stochastic.png (随机性采样)")
    print("模型权重: ddim_model_cifar.pth") 