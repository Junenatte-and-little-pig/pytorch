# -*- encoding: utf-8 -*-
import h5py
import torch


def main():
    points = torch.rand([4, 3], dtype=torch.float)
    with h5py.File('../data/p1ch3/mypoints.hdf5', 'w') as f:
        f.create_dataset('random', data=points.numpy())

    # with h5py.File('../data/p1ch3/ourpoints.hdf5','r') as f:
    #     dataset=f['coords']
    # this way will lead to an error that dataset gets wrong result
    # why?
    f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'r')
    dataset = f['coords']
    ourpoints = torch.from_numpy(dataset[()])  # instead of using value()
    print(ourpoints)


if __name__ == '__main__':
    main()
