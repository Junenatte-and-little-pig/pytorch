# -*- encoding: utf-8 -*-
import torch


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


def grad_fn(t_u,t_c,t_p,w,b):
    dloss_dtp=dloss_fn(t_p,t_c)
    dloss_dw=dloss_dtp*dmodel_dw(t_u,w,b)
    dloss_db=dloss_dtp*dmodel_db(t_u,w,b)
    return torch.stack([dloss_dw.sum(),dloss_db.sum()])


def gradient_decent(epoches,learning_rate,params,t_u,t_c):
    for epoch in range(epoches):
        w,b=params
        t_p=model(t_u,w,b)
        loss=loss_fn(t_p,t_c)
        grad=grad_fn(t_u,t_c,t_p,w,b)
        params=params-learning_rate*grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params


def main():
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    w = torch.ones(())
    b = torch.zeros(())
    gradient_decent(100,0.0001,torch.tensor([1.,0.]),t_u,t_c)


if __name__ == '__main__':
    main()
