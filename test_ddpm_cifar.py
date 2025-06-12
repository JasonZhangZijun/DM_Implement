# -*- coding: utf-8 -*-
"""
测试预训练的 DDPM CIFAR-10 模型
"""

import torch
import torchvision
import torchvision.transforms as T
import os
from models.ddpm import DDPM_Model

def test_cifar_ddpm(model_path="diffusion_model_cifar.pth", num_samples=16, output_path="ddpm_cifar_test_samples.png"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class FakeDataset(torch.utils.data.Dataset):
        def __getitem__(self, index):
            # 返回与模型输入匹配的随机张量
            return torch.randn(3, 32, 32), 0
        def __len__(self):
            return num_samples
            
    dataloader = torch.utils.data.DataLoader(FakeDataset(), batch_size=num_samples)

    model = DDPM_Model(dataloader, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, dataset_name="cifar")
        
    torchvision.utils.save_image(samples, output_path, nrow=4)

if __name__ == "__main__":
    test_cifar_ddpm()
