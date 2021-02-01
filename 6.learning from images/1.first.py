# -*- encoding: utf-8 -*-
from torchvision import datasets


def main():
    data_path='..\\data\\p1ch7'
    cifar10=datasets.CIFAR10(data_path,train=True,download=True)
    cifar10_val=datasets.CIFAR10(data_path,train=False,download=True)



if __name__ == '__main__':
    main()
