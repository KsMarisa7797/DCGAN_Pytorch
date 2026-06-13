import torch
import os

from Discriminative import D_Net
from Generator import G_Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = 'checkpoint_epoch_0650.pth'
ckpt_path = 'checkpoint/'+ckpt
save_dir = 'extracted_models'

gen = G_Net().to(device)
dis = D_Net().to(device)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
gen.load_state_dict(ckpt['gen'])
dis.load_state_dict(ckpt['dis'])

gen.eval()
dis.eval()

os.makedirs(save_dir, exist_ok=True)

gen_save_path = os.path.join(save_dir, 'generator.pth')
dis_save_path = os.path.join(save_dir, 'discriminator.pth')


torch.save(gen.state_dict(), gen_save_path)
torch.save(dis.state_dict(), dis_save_path)

print(f'生成器权重已保存到: {gen_save_path}')
print(f'判别器权重已保存到: {dis_save_path}')

