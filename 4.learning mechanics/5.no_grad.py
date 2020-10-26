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


def gradient_decent(epoches, optimizer, params, train_t_u, train_t_c, val_t_u,
                    val_t_c):
    for epoch in range(epoches):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad==False
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print('Epoch %d, Training Loss %f, Validating Loss %f' % (
            epoch, float(train_loss), float(val_loss)))
    return params


def main():
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)
    shuffled_indices = torch.randperm(n_samples)
    train_indices = shuffled_indices[:n_val]
    val_indices = shuffled_indices[n_val:]
    train_t_c = t_c[train_indices]
    train_t_u = t_u[train_indices]
    val_t_c = t_c[val_indices]
    val_t_u = t_u[val_indices]
    params = torch.tensor([1., 0.], requires_grad=True)
    # params.grad is None
    learning_rate = 1e-5
    optimizer = optim.SGD([params], lr=learning_rate)
    gradient_decent(1000, optimizer, params, train_t_u, train_t_c, val_t_u,
                    val_t_c)


if __name__ == '__main__':
    main()
