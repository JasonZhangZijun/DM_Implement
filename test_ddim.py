# DDIM模型简单测试
import torch
import torch.nn as nn
from models.ddim import DDIM_Model

# 创建一个简单的假数据加载器用于测试
class DummyDataLoader:
    def __init__(self):
        self.batch_size = 4
        
    def __iter__(self):
        # 生成一些假数据用于测试
        for i in range(2):  # 只生成2个批次用于测试
            fake_data = torch.randn(self.batch_size, 3, 32, 32)
            fake_labels = torch.randint(0, 10, (self.batch_size,))
            yield fake_data, fake_labels

def test_ddim():
    print("开始测试DDIM模型...")
    
    # 创建假数据加载器
    dummy_dataloader = DummyDataLoader()
    
    # 创建DDIM模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    ddim_model = DDIM_Model(dummy_dataloader, T=100, device=device)  # 使用较小的T进行测试
    
    # 测试前向过程
    print("\n测试前向过程...")
    x_0 = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 100, (2,)).to(device)
    noise = torch.randn_like(x_0)
    x_t = ddim_model.q_sample(x_0, t, noise)
    print(f"前向过程成功，输出形状: {x_t.shape}")
    
    # 测试损失计算
    print("\n测试损失计算...")
    loss = ddim_model.calculate_loss(x_0, t)
    print(f"损失计算成功，损失值: {loss.item():.4f}")
    
    # 测试DDIM采样步骤
    print("\n测试DDIM采样步骤...")
    ddim_model.unet.eval()
    with torch.no_grad():
        x_test = torch.randn(2, 3, 32, 32).to(device)
        t_tensor = torch.tensor([50, 50]).to(device)
        x_prev = ddim_model.ddim_step(x_test, t_tensor, 40, eta=0.0)
        print(f"DDIM采样步骤成功，输出形状: {x_prev.shape}")
    
    # 测试完整采样过程
    print("\n测试完整采样过程...")
    with torch.no_grad():
        samples = ddim_model.sample(2, ddim_steps=10, eta=0.0)
        print(f"完整采样成功，输出形状: {samples.shape}")
        print(f"样本数值范围: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    
    # 测试快速采样
    print("\n测试快速采样...")
    with torch.no_grad():
        fast_samples = ddim_model.fast_sample(2, ddim_steps=5, eta=0.0)
        print(f"快速采样成功，输出形状: {fast_samples.shape}")
    
    print("\n✅ 所有DDIM功能测试通过！")
    print("DDIM模型已成功实现，可以被外层训练代码调用。")

if __name__ == "__main__":
    test_ddim() 