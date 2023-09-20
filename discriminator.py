import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, features_d, img_channels):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=features_d, kernel_size=4, stride=2, padding=1), #Has bias
            nn.LeakyReLU(0.2),
            self._conv(in_ch=features_d, out_ch=features_d*2, kernel_size=4, stride=2, padding=1), #No bias
            self._conv(in_ch=features_d*2, out_ch=features_d*4, kernel_size=4, stride=2, padding=1),
            self._conv(in_ch=features_d * 4, out_ch=features_d * 8, kernel_size=4, stride=2, padding=1),
            self._conv(in_ch=features_d * 8, out_ch=1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid()   ### Since I am using autocast, I cannot use BCE but BCEWithLogitsLoss which
            # comes with it's own Sigmoid layer so I am removing this.
        )
    def _conv(self, in_ch, out_ch, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.disc(x)

if __name__ == "__main__":
    x = torch.randn((32, 3, 64, 64))
    gen = Discriminator(8, 3)
    print(gen(x).shape)