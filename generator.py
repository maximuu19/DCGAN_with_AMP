import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super().__init__()
        self.gnr = nn.Sequential(
            self._deconv(in_c=z_dim, out_c=features_g * 16, kernel_size=4, stride=1, padding=0),
            self._deconv(in_c=features_g * 16, out_c=features_g * 8, kernel_size=4, stride=2, padding=1),
            self._deconv(in_c=features_g * 8, out_c=features_g * 4, kernel_size=4, stride=2, padding=1),
            self._deconv(in_c=features_g * 4, out_c=features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features_g * 2, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def _deconv(self, in_c, out_c, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gnr(x)


if __name__ == "__main__":
    z_dim = 100
    x = torch.randn((32, z_dim, 1, 1))
    gen = Generator(z_dim, 3, 8)
    print(gen(x).shape)
