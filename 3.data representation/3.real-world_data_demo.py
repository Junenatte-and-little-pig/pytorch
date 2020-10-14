# -*- encoding: utf-8 -*-
import pandas as pd
import torch


def main():
    data_raw=pd.read_csv('../data/p1ch4/tabular-wine/winequality-white.csv',sep=';')
    # print(data)
    # print(data.shape)
    data_t=torch.from_numpy(data_raw.values)
    # print(data_t)
    data=data_t[:,:-1]
    # print(data)
    # print(data.shape)
    target_raw=data_t[:,-1].long()
    target=torch.zeros(target_raw.shape[0],10)
    target.scatter_(1,target_raw.unsqueeze(1),1.0)
    # print(target)
    # print(target.shape)
    data_mean=torch.mean(data,dim=0)
    data_var=torch.var(data,dim=0)
    data_normalized=(data-data_mean)/torch.sqrt(data_var)
    # print(data_normalized)

    bad_indexes=target_raw<=3




if __name__ == '__main__':
    main()
