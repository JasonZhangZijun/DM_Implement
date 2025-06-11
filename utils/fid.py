# FID计算工具 - 用于评估生成图片质量
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob
from scipy.linalg import sqrtm

class InceptionV3FeatureExtractor(nn.Module):
    """简化的Inception V3特征提取器"""
    def __init__(self):
        super().__init__()
        # 简化的特征提取网络，用于替代完整的Inception V3
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二层卷积块
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三层卷积块
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第四层卷积块
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_images_from_folder(folder_path, image_size=32, max_images=None):
    """从文件夹加载图片"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 支持多种图片格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
    
    if len(images) == 0:
        raise ValueError(f"在文件夹 {folder_path} 中没有找到有效的图片")
    
    return torch.stack(images)

def extract_features(images, feature_extractor, device, batch_size=32):
    """提取图片特征"""
    feature_extractor.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            batch_features = feature_extractor(batch)
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def calculate_fid(act1, act2):
    """计算FID分数"""
    # 计算均值和协方差矩阵
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # 计算均值差的平方和
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # 计算协方差矩阵乘积的平方根
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # 检查并修正平方根产生的复数
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算FID分数
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_fid_from_folders(real_folder, generated_folder, device=None, max_images=None):
    """从两个文件夹计算FID分数"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"加载真实图片从: {real_folder}")
    print(f"加载生成图片从: {generated_folder}")
    
    # 创建特征提取器
    feature_extractor = InceptionV3FeatureExtractor().to(device)
    
    # 加载图片
    real_images = load_images_from_folder(real_folder, max_images=max_images)
    generated_images = load_images_from_folder(generated_folder, max_images=max_images)
    
    print(f"加载了 {len(real_images)} 张真实图片")
    print(f"加载了 {len(generated_images)} 张生成图片")
    
    # 提取特征
    print("提取真实图片特征...")
    real_features = extract_features(real_images, feature_extractor, device)
    
    print("提取生成图片特征...")
    generated_features = extract_features(generated_images, feature_extractor, device)
    
    # 计算FID
    fid_score = calculate_fid(real_features, generated_features)
    
    return fid_score

def evaluate_generated_images(generated_folder, real_folder=None, device=None, max_images=1000):
    """评估生成图片的FID分数"""
    if real_folder is None:
        # 如果没有提供真实图片文件夹，创建一些随机特征作为参考
        print("警告: 没有提供真实图片文件夹，使用随机特征作为参考")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        feature_extractor = InceptionV3FeatureExtractor().to(device)
        generated_images = load_images_from_folder(generated_folder, max_images=max_images)
        generated_features = extract_features(generated_images, feature_extractor, device)
        
        # 创建随机参考特征
        real_features = np.random.randn(len(generated_features), generated_features.shape[1])
        fid_score = calculate_fid(real_features, generated_features)
    else:
        fid_score = calculate_fid_from_folders(real_folder, generated_folder, device, max_images)
    
    return fid_score

# 测试代码
if __name__ == "__main__":
    # 创建一些测试数据
    print("测试FID计算...")
    
    # 创建随机激活值进行测试
    act1 = np.random.randn(100, 2048)
    act2 = np.random.randn(100, 2048)
    
    # 测试相同数据的FID（应该接近0）
    fid = calculate_fid(act1, act1)
    print(f'FID (相同数据): {fid:.3f}')
    
    # 测试不同数据的FID
    fid = calculate_fid(act1, act2)
    print(f'FID (不同数据): {fid:.3f}')
    
    print("FID计算工具已准备就绪！")