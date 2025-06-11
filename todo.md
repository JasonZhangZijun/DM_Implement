你已经完成了 `ddpm.py`, `unet.py`, `diffusion.py` 并能在 CIFAR-10 上进行 unconditional 训练与采样，接下来可以基于 proposal 的 roadmap，把整个项目 modular 且系统地拓展为 **多任务 diffusion pipeline**。下面是具体的建议，按阶段逐步进行，每个阶段我会给出：

* 🎯 **目标任务**
* 🧱 **模块与接口建议**
* 🛠 **关键实现点**
* 🔍 **实验/评估建议**

---

## ✅ Phase 1（你已完成）

🎯 **Unconditional Generation on CIFAR-10**
你已经实现了基础 DDPM 训练与采样，这部分你只需要做一个简单清理：

🧱 **建议整理模块结构：**

```bash
project/
│
├── models/
│   ├── ddpm.py       # 包含forward & reverse过程定义
│   ├── unet.py       # 网络架构
│
├── diffusion.py      # 训练/采样主脚本
├── utils.py          # 保存图片、调度器、采样器、mask工具等
├── configs.py        # 超参数管理
└── README.md         # 简要介绍
```

🛠 **你可以补充的点：**

* 支持 FID 指标评估（自己写，不能用`torchmetrics`）；
* 将 sampling 拆出来成一个 `sample.py`，保持训练主函数简洁；
* 实现 DDIM sampling 作为加速选项（参考 DDIM 的更新规则）；
* 日志输出加入 `wandb` 或简单 `matplotlib` 可视化。

---

## 🧩 Phase 2: Image Inpainting

🎯 **任务**：训练 DDPM 在已知部分图像的情况下补全缺失区域。

🧱 **新增模块建议：**

* `utils.masking.py`：提供 center, box, irregular mask 生成工具；
* `models/inpaint_ddpm.py`：修改 `DDPM_Model` 支持 mask-aware 训练；
* `configs_inpaint.py`：设置 inpainting 的配置；
* `inpaint.py`：单独的训练脚本（类似 `diffusion.py`）

🛠 **关键实现点：**

1. 输入图像 → 加入 mask → 给 UNet 输入 `(masked_image, mask)`；
2. 训练时 loss 仍为预测 noise 的 MSE；
3. mask 作为一个额外的通道连接到 UNet；
4. sample 时使用：

   ```python
   x_t[mask==0] = known_region[mask==0]
   ```

   每一步都保持已知区域不变。

🔍 **评估建议：**

* 目视检查（保存可视化样例）
* `PSNR`, `SSIM` （这两个指标你可以手写实现）

---

## 🎨 Phase 3（Optional）: Text / Class Conditional Generation

🎯 **任务**：通过 one-hot 向量或 embedding 来生成特定类别的图像。

🧱 **新增模块建议：**

* `models/conditional_unet.py`：UNet 支持 class embedding 输入；
* `conditional_diffusion.py`：包含 one-hot label → feature concat 的逻辑。

🛠 **关键实现点：**

1. 使用 CIFAR-10 标签 (0–9)，转为 one-hot；
2. 方案 A（简单）：embedding + concat 到 UNet 输入；
3. 方案 B（进阶）：使用 FiLM 层调节 UNet 中间层；
4. loss 与 unconditional 一致。

🔍 **评估建议：**

* 每类生成若干图片；
* 使用分类器评估生成图像的类别准确率。

---

## 🖼 Phase 4（Optional）: Super-Resolution & Style Transfer

### 🎯 超分辨率

🧱 模块建议：

* `lowres_dataset.py`：输入 image → downsample（用 `torch.nn.functional.interpolate`）
* `sr_ddpm.py`：修改采样流程从低分图开始逆扩散。

🛠 核心是将 `x_0` 替换为 upsampled 图像（作为条件输入），网络学会生成高分图像。

---

### 🎯 风格迁移

🧱 模块建议：

* `style_encoder.py`：抽取 style image 的 embedding；
* `style_cond_unet.py`：在 UNet 中融合 style embedding（如 AdaIN, FiLM）；

🛠 用法和 text-conditional 类似，但输入变为图片 → style vector。

---

## 📦 总结阶段性任务安排

| 阶段      | 模块目标                  | 是否必须   |
| ------- | --------------------- | ------ |
| Phase 1 | DDPM + CIFAR-10 训练/采样 | ✅ 已完成  |
| Phase 2 | 图像 inpainting         | ✅ 强烈推荐 |
| Phase 3 | Class conditional 生成  | 🔸 可选  |
| Phase 4 | 超分辨率 & 风格迁移           | 🔸 可选  |

---

## 🧭 建议的模块拆分清单

你可以将代码整理为如下结构：

```
project/
├── models/
│   ├── unet.py
│   ├── conditional_unet.py
│   ├── ddpm.py
│   ├── inpaint_ddpm.py
│   ├── style_cond_unet.py
│
├── utils/
│   ├── masks.py
│   ├── sampling.py
│   ├── fid_score.py
│   ├── losses.py
│
├── diffusion.py
├── sample.py
├── inpaint.py
├── conditional_diffusion.py
├── sr_diffusion.py
├── config.py
└── README.md
```

---

如果你想，我可以为你：

* 写好 inpainting 的数据处理逻辑和 mask 工具；
* 构建一个 class conditional 的 UNet；
* 解释 DDIM 如何推导并实现；
* 设计 `config.py` 管理不同任务的超参数；

告诉我你希望我从哪部分开始。
