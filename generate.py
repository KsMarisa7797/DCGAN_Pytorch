import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Generator import G_Net
from Discriminative import D_Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = 'checkpoint'
extracted_models_path = 'extracted_models'

gen = G_Net().to(device)
gen.load_state_dict(torch.load(checkpoint_path+'/generator.pth', map_location=device, weights_only=True))
gen.eval()

dis = D_Net().to(device)
dis.load_state_dict(torch.load(checkpoint_path+'/discriminator.pth', map_location=device, weights_only=True))
dis.eval()

num_images = 64
with torch.no_grad():
    noise = torch.randn(num_images, 128, 1, 1, device=device)
    fake_images = gen(noise)
    scores = dis(fake_images).squeeze()

fake_images_np = fake_images.permute(0, 2, 3, 1).cpu().numpy()
scores_np = scores.cpu().numpy()

os.makedirs('generated', exist_ok=True)

# 保存全部64张图 (8x8)
cols = 8
rows = num_images // cols
fig = plt.figure(figsize=(cols * 2, rows * 2))
for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow((fake_images_np[i] + 1) / 2)
    plt.axis('off')
plt.tight_layout()
plt.savefig('generated/all_64.png')
plt.close(fig)
print(f'64 images saved to generated/all_64.png')

# 用D_Net分数选top 9 (3x3)
top9_idx = scores_np.argsort()[-9:][::-1]
fig = plt.figure(figsize=(6, 6))
for j, idx in enumerate(top9_idx):
    plt.subplot(3, 3, j + 1)
    plt.imshow((fake_images_np[idx] + 1) / 2)
    plt.axis('off')
    plt.title(f'score: {scores_np[idx]:.2f}', fontsize=8)
plt.tight_layout()
plt.savefig('generated/top9.png')
plt.close(fig)
print(f'Top 9 images saved to generated/top9.png')
print(f'Scores range: [{scores_np.min():.2f}, {scores_np.max():.2f}]')
