# -*- encoding: utf-8 -*-
import torch
import torch.optim as optim


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_c, t_u):
    diff = (t_c - t_u) ** 2
    return diff.mean()


def dloss_fn(t_p, t_c):
    return 2 * (t_p - t_c) / t_c.size(0)


def dmodel_dw(t_u, w, b):
    return t_u


def dmodel_db(t_u, w, b):
    return 1.0


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


def gradient_decent(epoches, optimizer, params, t_u, t_c):
    for epoch in range(epoches):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


def main():
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    params = torch.tensor([1., 0.], requires_grad=True)
    # params.grad is None
    learning_rate = 1e-5
    optimizer = optim.SGD([params], lr=learning_rate)
    gradient_decent(1000, optimizer, params, t_u, t_c)


if __name__ == '__main__':
    main()
