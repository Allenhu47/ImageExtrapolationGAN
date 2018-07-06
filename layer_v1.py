#coding=utf-8
import torch
import torch.nn as nn

class CompletionNet(nn.Module):
    def __init__(self):
        super(CompletionNet, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
                #nn.BatchNorm2d(64),
                nn.ELU(True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
                #nn.BatchNorm2d(128),
                nn.ELU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.BatchNorm2d(128),
                nn.ELU(True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),

                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=4, dilation=4, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=8, dilation=8, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=16, dilation=16, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),

                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.BatchNorm2d(256),
                nn.ELU(True),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
                #nn.BatchNorm2d(128),
                nn.ELU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.BatchNorm2d(128),
                nn.ELU(True),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
                #nn.BatchNorm2d(64),
                nn.ELU(True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.BatchNorm2d(32),
                nn.ELU(True),
                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Tanh(),
                # nn.ELU(True)
            )
    def forward(self, input_data):
        # print(input_data)
        return self.main(input_data)


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=3, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        #utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)


        return x
