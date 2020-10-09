# -*- encoding: utf-8 -*-
import os

import cv2 as cv
import torch


def main():
    # img_np=cv.imread('../data/p1ch4/image-dog/bobby.jpg')
    # print(img_np.shape) # H * W * C
    # img=torch.from_numpy(img_np)
    # img_t=img.permute(2,0,1) # share the same storage
    # print(img_t.shape) # C * H * W

    data_dir = '../data/p1ch4/image-cats'
    filenames = [name for name in os.listdir(data_dir) if
                 os.path.splitext(name)[-1] == '.png']
    batch_size = len(filenames)
    batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
    for i, filename in enumerate(filenames):
        img_np = cv.imread(os.path.join(data_dir, filename))
        img_t = torch.from_numpy(img_np)
        img_t = img_t.permute(2, 0, 1)
        img_t = img_t[:3]  # make sure img_t only have 3 channels
        batch[i] = img_t
    print(batch)
    # normalization 1
    batch = batch.float()
    batch /= 255.0
    print(batch)  # 0~1

    # normalization 2
    n_channels = batch.shape[1]
    for c in range(n_channels):
        mean = torch.mean(batch[:, c])
        std = torch.std(batch[:, c])
        batch[:, c] = (batch[:, c] - mean) / std
    print(batch)  # mean 0


if __name__ == '__main__':
    main()
