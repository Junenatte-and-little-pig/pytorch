# -*- encoding: utf-8 -*-
import torch


def main():
    points = torch.tensor([[4., 2.], [2., 3.], [9., 7.]], dtype=torch.float)
    points_gpu = torch.tensor([[4., 2.], [2., 3.], [9., 7.]], device='cuda')
    points_gpu_another = points.to(device='cuda')  # also can get from cuda()
    points_cpu = points_gpu.to(device='cpu')  # also can get from cpu()


if __name__ == '__main__':
    main()
