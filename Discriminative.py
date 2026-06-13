import torch
import torch.nn as nn

class MinibatchStdDev(nn.Module):
    def forward(self, x):
        batch,C,H,W = x.shape
        std_map = x.std(dim=0,keepdim=False)
        std_val = std_map.mean()
        std_channel = std_val.expand(batch, 1, H, W)
        return torch.cat([x,std_channel],1)


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.minibatch_std = MinibatchStdDev()
        self.final_conv = nn.Conv2d(513, 1, 4, 1, 0, bias=False)

    def forward(self,x,return_features=False):
        x = self.features(x)
        feat = x
        x = self.minibatch_std(x)
        x = self.final_conv(x)
        if return_features:
            return x,feat
        return x

    def d_weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.02)

if __name__ == '__main__':
    d_net = D_Net()
    input = torch.ones(1, 3, 64, 64)
    output = d_net(input)
    print(output.shape)
