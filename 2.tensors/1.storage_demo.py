# -*- encoding: utf-8 -*-
import torch


def main():
    # as usually used in experiments, dtype is always set as torch.float(32) or torch.int64(a.k.a. torch.long)
    points = torch.tensor([[4., 2.], [2., 3.], [9., 7.]], dtype=torch.float)
    storage = points.storage()
    print(storage)
    # as given an index [a0, a1, a2, ..., an], the element to be referred to is stored at
    # storage_offset + stride[0] * a0 + stride[1] * a1 + stride[2] * a2 + ... + stride[n] * an
    # from the start
    print(points.storage_offset())
    print(points.stride())

    points_clone = points.clone()
    points_t = points.t()
    # it shows that transpose operation just share the same storage with the origin tensor
    print(points.storage() == points_t.storage())
    # it change the stride to get the same transpose effect
    print(points_t.stride())

    # as the tensor has been transposed, the storage is not contiguous for the new tensor
    # it will make some operation as view() become invalid
    # so use contiguous() before view() if transposed
    print(points.is_contiguous())
    print(points_t.is_contiguous())
    print(points_t.contiguous().view(2, 3))


if __name__ == '__main__':
    main()
