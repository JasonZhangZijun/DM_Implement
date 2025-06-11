# diffusion model
from models.unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.ddpm import DDPM_Model
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

### 1. prepare the datasets cifar10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
transform = T.Compose([
    T.CenterCrop(178),        # CelebA 的人脸裁剪
    T.Resize((64, 64)),       # or 32×32 视网络而定
    T.ToTensor(),
    T.Normalize((0.5,)*3, (0.5,)*3),
])

dataset = ImageFolder(root='./data/celeba', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 2. train the diffusion model
if __name__ == "__main__":
    diffusion_model = DDPM_Model(dataloader)
    diffusion_model.train_model(num_epochs=100)
    # 保存模型状态字典而不是整个模型
    torch.save(diffusion_model.state_dict(), "diffusion_model_celeba.pth")
    # 创建新模型并加载状态字典
    diffusion_model = DDPM_Model(dataloader)
    diffusion_model.load_state_dict(torch.load("diffusion_model_celeba.pth"))
    
    # 生成一些样本并保存
    x_0 = diffusion_model.sample(10)
    torchvision.utils.save_image(x_0, "sample.png")
    
    # 评估模型
    print("开始评估模型...")
    fid_score = diffusion_model.evaluate(num_samples=1000)
    print(f"最终 FID 分数: {fid_score:.2f}")