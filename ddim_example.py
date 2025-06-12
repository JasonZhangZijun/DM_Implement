# DDIM模型示例
from models.unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.ddim import DDIM_Model

### 1. 准备CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 2. 训练DDIM模型
if __name__ == "__main__":
    # 创建DDIM模型
    ddim_model = DDIM_Model(dataloader, device=device)
    
    # 训练模型
    # print("开始训练DDIM模型...")
    # ddim_model.train_model(num_epochs=50, lr=1e-4, save_path="ddim_model.pth")
    
    # 保存和加载模型
    # torch.save(ddim_model.state_dict(), "ddim_model.pth")
    
    # 创建新模型并加载状态字典
    ddim_model_loaded = DDIM_Model(dataloader, device=device)
    ddim_model_loaded.load_state_dict(torch.load("diffusion_model_cifar.pth"))
    # ddim_model_loaded.load_state_dict(torch.load("ddim_model.pth"))
    
    # 使用标准DDIM采样生成样本（50步）
    print("使用标准DDIM采样生成样本...")
    x_standard = ddim_model_loaded.sample(8, ddim_steps=100, eta=0.0)
    torchvision.utils.save_image(x_standard, "ddim_sample_standard.png", normalize=True)
    
    # 使用快速DDIM采样生成样本（10步）
    print("使用快速DDIM采样生成样本...")
    x_fast = ddim_model_loaded.fast_sample(8, ddim_steps=300, eta=0.0)
    torchvision.utils.save_image(x_fast, "ddim_sample_fast.png", normalize=True)
    
    # 使用随机性采样（eta > 0）
    print("使用随机性DDIM采样生成样本...")
    x_stochastic = ddim_model_loaded.sample(8, ddim_steps=1000, eta=0.5)
    torchvision.utils.save_image(x_stochastic, "ddim_sample_stochastic.png", normalize=True)
    
    # 评估模型
    print("开始评估DDIM模型...")
    fid_score = ddim_model_loaded.evaluate(num_samples=1000)
    print(f"最终 FID 分数: {fid_score:.2f}")
    
    print("DDIM模型训练和采样完成！")
    print("生成的图像保存为:")
    print("- ddim_sample_standard.png (标准50步采样)")
    print("- ddim_sample_fast.png (快速10步采样)")
    print("- ddim_sample_stochastic.png (随机性采样)") 