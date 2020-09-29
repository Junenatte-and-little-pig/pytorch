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
    img=Image.open("../data/p1ch2/bobby.jpg")
    img.show()
    img_t=preprocess(img)
    batch_t=torch.unsqueeze(img_t,0)
    resnet.eval()
    out=resnet(batch_t)
    print(out)

    with open("../data/p1ch2/imagenet_classes.txt") as f:
        label=[line.strip() for line in f.readlines()]
    _,index=torch.max(out,1)
    percentage=torch.nn.functional.softmax(out,dim=1)[0]*100
    print(label[index[0]])
    print(percentage[index[0]].item())

    _,indices=torch.sort(out,descending=True)
    print([(label[idx],percentage[idx].item()) for idx in indices[0][:5]])



if __name__ == '__main__':
    main()
