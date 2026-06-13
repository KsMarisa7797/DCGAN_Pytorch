import torch
import torch.nn as nn

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.modul = nn.Sequential(
            nn.ConvTranspose2d(128,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.modul(x)
        return x

    def g_weight_init(self, m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)

if __name__ == '__main__':
    g_net = G_Net()
    input = torch.ones(1, 128, 1, 1)
    output = g_net(input)
    print(output.shape)

