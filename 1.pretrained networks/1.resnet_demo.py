# -*- encoding: utf-8 -*-
from torchvision import models,transforms
from PIL import Image
import torch


def main():
    resnet=models.resnet101(pretrained=True)
    preprocess=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    img=Image.open()
    img_t=preprocess(img)
    batch_t=torch.unsqueeze(img_t,0)
    resnet.eval()
    out=resnet(batch_t)
    print(out)



if __name__ == '__main__':
    main()
