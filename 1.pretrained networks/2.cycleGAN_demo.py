# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        # ReflectionPad2d: 2维的反射填充
        # Conv2d: 2维卷积层，其中padding会使用bias对输入数据进行提前填充
        # InstanceNorm2d: 对每个channel内的实例进行归一化
        # ReLU: 激活函数
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        assert (n_blocks >= 0)
        super(ResNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            multi = 2 ** i
            model += [
                nn.Conv2d(ngf * multi, ngf * multi * 2, kernel_size=3, stride=2,
                          padding=1, bias=True),
                nn.InstanceNorm2d(ngf * multi * 2),
                nn.ReLU(True)]

        multi = 2 * n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * multi)]

        for i in range(n_downsampling):
            multi = 2 ** (n_downsampling - i)
            # ConvTranspose: Conv的逆过程
            model += [
                nn.ConvTranspose2d(ngf * multi, ngf * multi // 2, kernel_size=3,
                                   stride=2, padding=1, output_padding=1,
                                   bias=True),
                nn.InstanceNorm2d(ngf * multi // 2),
                nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def main():
    resNetGenerator = ResNetGenerator()
    # print(resNetGenerator)
    model_path = "../data/p1ch2/horse2zebra_0.4.0.pth"
    model_data = torch.load(model_path)
    resNetGenerator.load_state_dict(model_data)
    resNetGenerator.eval()
    preprocess = transforms.Compose(
        [transforms.Resize(256), transforms.ToTensor()])
    img = Image.open('../data/p1ch2/horse.jpg')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_out = resNetGenerator(batch_t)
    out_t = (batch_out.data.squeeze() + 1.0) / 2.0
    out_img = transforms.ToPILImage()(out_t)
    out_img.save('../data/p1ch2/zebra.jpg')
    out_img.show()


if __name__ == '__main__':
    main()
