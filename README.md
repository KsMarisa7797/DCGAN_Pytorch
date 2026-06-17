# DCGAN_PyTorch — 深度卷积生成对抗网络

基于PyTorch实现的DCGAN(Deep Convolutional GAN)，用于生成 **64×64** 的动漫图像。该项目采用多种现代 GAN 训练技巧，提供完整的 **训练 → 断点续训 → 生成** 流程。
训练数据集来自于 https://www.kaggle.com/datasets/soumikrakshit/anime-faces

1500轮次训练的生成图像

<img width="60%" alt="all_64" src="https://github.com/user-attachments/assets/630768c8-048c-48a7-bf15-7df400b46462" />

判别器和生成器的loss折线图
*本项目使用了断点训练下图经过了多次调参经供参`以实际训练为准`*

<img width="1000" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/6962dc6c-507a-4f13-9d1f-b94e4b602dc4" />


---

## 项目结构

```
DCGAN_Pytorch/
├── Generator.py          # 生成器网络定义
├── Discriminative.py     # 判别器网络定义（含 MinibatchStdDev）
├── train.py              # 训练主脚本（含数据加载、断点续训、loss 曲线保存）
├── generate.py           # 加载模型并生成图像（输出 64 张 + top 9）
├── loadcheckpoint.py     # 从 checkpoint 中提取独立的模型权重
├── test.py               # 数据预览脚本
├── data/                 # 训练数据（*.png 图片）
├── checkpoint/           # 训练 checkpoint（每 50 epoch 保存）
├── output/               # 训练过程中每个 epoch 的生成效果图 + loss 曲线
├── generated/            # 最终生成结果（all_64.png / top9.png）
└── extracted_models/     # 提取出的独立模型权重文件
```

---

## 网络结构

### 生成器(`Generator.py`)

| 层 | 类型 |  输入 - 输出 | Kernel/Stride/Padding |
|---|---|---|---|
| 1 | ConvTranspose2d + BN + ReLU| 128 - 512, 4x4 | 4/1/0 |
| 2 | ConvTranspose2d + BN + ReLU| 512 - 256, 8x8 | 4/2/1 |
| 3 | ConvTranspose2d + BN + ReLU| 256 - 128, 16x16| 4/2/1 |
| 4 | ConvTranspose2d + BN + ReLU| 128 - 64, 32x32 | 4/2/1 |
| 5 | ConvTranspose2d + Tanh | 64 - 3, 64x64 | 4/2/1 |

- 输入：`[batch, 128, 1, 1]` 的随机噪声
- 输出：`[batch, 3, 64, 64]` 的 RGB 图像（像素值 ∈ [-1, 1]）

### 判别器(`Discriminative.py`)

| 层 | 类型 | 输入 → 输出 | Kernel/Stride/Padding |
|---|---|---|---|
| 1 | Conv2d + LeakyReLU(0.2) | 3 → 64 | 4/2/1 |
| 2 | Conv2d + BN + LeakyReLU(0.2) | 64 → 128 | 4/2/1 |
| 3 | Conv2d + BN + LeakyReLU(0.2) | 128 → 256 | 4/2/1 |
| 4 | Conv2d + BN + LeakyReLU(0.2) | 256 → 512 | 4/2/1 |
| — | **MinibatchStdDev** | 512 → 513 | 拼接 minibatch 标准差 |
| 5 | Conv2d | 513 → 1 | 4/1/0 |

- 输出：`[batch, 1, 1, 1]` 的真/假判别 logits

---

## 关键特性

| 技巧 | 说明 |
|---|---|
| **Minibatch Standard Deviation** | 在判别器末端拼接 minibatch 标准差通道，提升生成多样性，缓解 mode collapse |
| **Label Smoothing** | 真实样本标签设为 0.9（而非 1.0），缓解判别器过拟合 |
| **输入噪声注入** | 对判别器真假输入均添加 σ=0.03 高斯噪声，提升鲁棒性 |
| **Feature Matching Loss** | 生成器损失中加入真假中间特征的 MSE（权重 0.02），稳定训练 |
| **混合精度训练 (AMP)** | 使用 `torch.amp.GradScaler`，加速训练并降低显存占用 |
| **断点续训** | 每 50 epoch 保存完整 checkpoint（模型 + 优化器 + scaler + loss），可无缝恢复 |
| **权重初始化** | 卷积/转置卷积权重按 N(0, 0.02²) 初始化 |

---

## 如何运行

### 1.准备训练数据集

将上文提到的训练数据集的data文件夹解压到项目根目录

### 2. 训练模型

运行`train.py`，`output/`中会记录每一轮次生成的8张图像方便观察生成质量和调试代码，如果你认为这些图片占用设备硬盘的存储空间可以将`train.py`中的`generate_and_save_images` 注释

训练配置（可在 `train.py` 中修改）：
- 总 Epoch：1500
- Batch Size：256
- 学习率：G = 1e-4，D = 2e-4
- Checkpoint 保存间隔：50 epoch

训练过程中会在 `output/` 目录下每个 epoch 保存一张生成效果预览图，训练结束后自动保存 loss 曲线。

### 3. 断点续训

训练中断后直接再次运行 `python train.py`，脚本会自动找到最新的 checkpoint 并恢复训练。

### 4. 生成图像

训练完成后运行`generate.py`

会生成：
- `generated/all_64.png` — 64 张随机生成图像（8×8 网格）
- `generated/top9.png` — 判别器评分最高的 9 张图像（3×3 网格）

### 5. 提取模型权重

如果你对某个`checkpoint`感兴趣可以在`loadcheckpoint.py`中提取独立的 `generator.pth` 和 `discriminator.pth`






