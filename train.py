import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch.utils import data

from Discriminative import D_Net
from Generator import G_Net

torch.backends.cudnn.benchmark = True


"""加载训练数据图片并且预处理"""
class Face_dataset(data.Dataset):
    def __init__(self, imgs_path, transform):
        self.imgs_path = imgs_path
        self.transform = transform

    def __getitem__(self, index):
        imgs_path = self.imgs_path[index]
        pil_img = Image.open(imgs_path)
        pil_img = self.transform(pil_img)
        return pil_img

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.imgs_path)

def generate_and_save_images(model, epoch, test_input):
    os.makedirs('output', exist_ok=True)
    predictions = model(test_input).permute(0, 2, 3, 1).cpu().numpy()
    fig = plt.figure(figsize=(20, 20))
    for i in range(predictions.shape[0]):
        plt.subplot(1, 8, i + 1)
        plt.imshow((predictions[i] + 1) / 2)
        plt.axis('off')
    plt.savefig(f'output/epoch_{epoch:04d}.png')
    plt.close(fig)


if __name__ == '__main__':
    #数据准备
    imgs_path = glob.glob('./data/*.png')# 获取所有png图片路径

    # 图片预处理：转为Tensor，并归一化到[-1,1]（mean=0.5, std=0.5 可实现 [0,1]->[-1,1]）
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),      # 自动将PIL图片[0,255]缩放到[0,1]的FloatTensor
        transforms.Normalize(mean=0.5, std=0.5), # (x-0.5)/0.5 => 范围变为[-1,1]
    ])

    dataset = Face_dataset(imgs_path, img_transform)

    BATCH_SIZE = 256
    dataloader = data.DataLoader(dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,              # 每个epoch打乱数据
                                 num_workers=8,             # 使用8个子进程加载数据
                                 pin_memory=True,           # 将数据锁页在内存，加速GPU拷贝
                                 prefetch_factor=4,         # 每个worker预取4个batch
                                 persistent_workers=True,   # 保持worker进程存活，减少创建开销
                                 )

    #初始化模型和训练组件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = G_Net().to(device)
    dis = D_Net().to(device)
    # gen.apply(gen.g_weight_init)
    # dis.apply(dis.d_weight_init)

    # 损失函数：二分类交叉熵（内部自带Sigmoid），同时适用于真/假判别
    loss_func = torch.nn.BCEWithLogitsLoss()

    # Adam优化器，β1=0.5 有助于GAN训练稳定
    d_optimizer = torch.optim.Adam(dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # 混合精度训练的梯度缩放器，可加速训练并减少显存占用
    scaler = torch.amp.GradScaler('cuda')

    #检查点加载，断点续训
    ckpt_dir = 'checkpoint'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = sorted(glob.glob(f'{ckpt_dir}/checkpoint_epoch_*.pth'))
    ckpt_path = ckpt_files[-1] if ckpt_files else None
    start_epoch = 0

    D_loss = []     # 记录每个epoch的平均判别器损失
    G_loss = []     # 记录每个epoch的平均生成器损失

    """加载checkpoint"""
    if ckpt_path is not None:
        # 加载模型、优化器、scaler状态以及历史loss和epoch信息
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        gen.load_state_dict(ckpt['gen'])
        dis.load_state_dict(ckpt['dis'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.param_groups[0]['lr'] = 2e-4  # 覆盖checkpoint中旧LR，保持D_G平衡
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch']+1
        D_loss = ckpt['D_loss']
        G_loss = ckpt['G_loss']
        print(f'Loaded checkpoint, resuming from epoch {start_epoch}')
    else:
        gen.apply(gen.g_weight_init)
        dis.apply(dis.d_weight_init)


    test_input = torch.randn(8, 128, 1, 1, device=device)
    total_epochs = 1500

    """训练循环"""
    for epoch in range(start_epoch,total_epochs):
        D_epoch_loss = 0
        G_epoch_loss = 0
        count = len(dataloader)

        for step, img in enumerate(dataloader):
            img = img.to(device)
            size = img.shape[0]
            #Discriminator，判别器训练两次保持在对抗中处于优势状态

            d_optimizer.zero_grad()
            random_seed = torch.randn(size, 128, 1, 1, device=device)
            with torch.amp.autocast('cuda'):
                img_noisy = img + torch.randn_like(img) * 0.03
                real_output = dis(img_noisy)
                # 判别器对真实图片的损失：希望输出接近0.9标签平滑，减轻过拟合
                d_real_loss = loss_func(real_output, torch.full_like(real_output, 0.9, device=device))
                generated_img = gen(random_seed)
                fake_noisy = generated_img.detach() + torch.randn_like(generated_img) * 0.03        #加入噪声，提高判别器鲁棒性
                fake_output_d = dis(fake_noisy)
                # 判别器对假图片的损失：希望输出接近0
                d_fake_loss = loss_func(fake_output_d, torch.zeros_like(fake_output_d, device=device))
                disc_loss = d_real_loss + d_fake_loss

                # 缩放判别器损失并反向传播，更新判别器参数
            scaler.scale(disc_loss).backward()
            scaler.step(d_optimizer)


            # 训练生成器
            g_optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # 让判别器对生成图片进行判断，同时输出中间层特征用于特征匹配损失
                fake_output_g, fake_feat = dis(generated_img, return_features=True)
                # 对抗损失，希望判别器将假图判定为真
                adv_loss = loss_func(fake_output_g, torch.ones_like(fake_output_g,device=device))
                _,real_feat = dis(img, return_features=True)
                fm_loss = torch.nn.functional.mse_loss(fake_feat, real_feat.detach())
                gen_loss = adv_loss + 0.02 * fm_loss

            scaler.scale(gen_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()

            with torch.no_grad():
                D_epoch_loss += disc_loss.item()
                G_epoch_loss += gen_loss.item()

            if step % 10 == 0:
                print(f'Epoch [{epoch + 1}/{total_epochs}] Step [{step}/{len(dataloader)}] '
                      f'D_loss: {disc_loss.item():.4f} G_loss: {gen_loss.item():.4f}')

        with torch.no_grad():
            D_epoch_loss /= count
            G_epoch_loss /= count
            D_loss.append(D_epoch_loss)
            G_loss.append(G_epoch_loss)
            generate_and_save_images(gen, epoch, test_input)
            print('Epoch:', epoch + 1)

        if (epoch + 1) % 50 == 0:  # 每50轮保存一次
            torch.save({
                'gen': gen.state_dict(),
                'dis': dis.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'D_loss': D_loss,
                'G_loss': G_loss,
            }, f'{ckpt_dir}/checkpoint_epoch_{epoch + 1:04d}.pth')
            print(f'Checkpoint saved at epoch {epoch + 1}')

    torch.save({
        'gen': gen.state_dict(),
        'dis': dis.state_dict(),
        'd_optimizer': d_optimizer.state_dict(),
        'g_optimizer': g_optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': total_epochs - 1,
        'D_loss': D_loss,
        'G_loss': G_loss,
    }, f'{ckpt_dir}/checkpoint_epoch_{total_epochs:04d}.pth')
    torch.save(gen.state_dict(), f'{ckpt_dir}/generator.pth')
    torch.save(dis.state_dict(), f'{ckpt_dir}/discriminator.pth')
    print('Final checkpoint saved')

    # 保存 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(D_loss, label='D_loss', color='blue')
    plt.plot(G_loss, label='G_loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator and Generator Loss')
    plt.savefig('output/loss_curve.png')
    plt.close()
    print('Loss curve saved to output/loss_curve.png')

