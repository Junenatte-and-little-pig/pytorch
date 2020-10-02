# -*- encoding: utf-8 -*-
import torch


def main():
    mypoints = torch.rand([4, 3])
    # points=torch.load('../data/p1ch3/ourpoints.t')
    with open('../data/p1ch3/ourpoints.t', 'rb') as f:
        points = torch.load(f)
    print(points)

    with open('../data/p1ch3/mypoints.t', 'wb') as f:
        torch.save(mypoints, f)


if __name__ == '__main__':
    main()
