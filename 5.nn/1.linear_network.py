# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim


def train(epoches, optimizer, model, loss_fn, train_t_u, train_t_c, val_t_u,
          val_t_c):
    for epoch in range(epoches):
        train_t_p = model(train_t_u)
        train_loss = loss_fn(train_t_p, train_t_c)
        val_t_p = model(val_t_u)
        val_loss = loss_fn(val_t_p, val_t_c)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print('Epoch %d, Training Loss %f, Validating Loss %f' % (
            epoch, float(train_loss), float(val_loss)))


def main():
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c).unsqueeze(1)
    t_u = torch.tensor(t_u).unsqueeze(1)

    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]

    t_u_train = t_u[train_indices]
    t_c_train = t_c[train_indices]

    t_u_val = t_u[val_indices]
    t_c_val = t_c[val_indices]

    t_un_train = 0.1 * t_u_train
    t_un_val = 0.1 * t_u_val

    linear_model = nn.Linear(1, 1)
    # linear_model(t_un_train) # use __call__() rather than forward()
    optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)
    train(2000, optimizer, linear_model, nn.MSELoss(), t_un_train, t_c_train,
          t_un_val, t_c_val)
    print(linear_model.weight, linear_model.bias)


if __name__ == '__main__':
    main()
