# -*- encoding: utf-8 -*-
import pandas as pd
import torch
import numpy as np


def main():
    bike_raw = pd.read_csv('../data/p1ch4/bike-sharing-dataset/hour-fixed.csv',
                           dtype=np.float32,
                           converters={1: lambda x: float(x[8:10])})
    bike = torch.from_numpy(bike_raw.values)
    # print(bike)
    # print(bike.shape)
    # -1 makes the computer to calculate the shape
    daily_bike = bike.view(-1, 24, bike.shape[1])
    daily_bike.transpose_(1, 2)
    print(daily_bike.shape)

    # first_day=bike[:24].long()
    # weather_onehot=torch.zeros(first_day.shape[0],4)
    # weather_onehot.scatter_(1,first_day[:,9].unsqueeze(1).long()-1,1.0)
    # print(weather_onehot)

    daily_weather_onehot = torch.zeros(daily_bike.shape[0], 4,
                                       daily_bike.shape[2])
    daily_weather_onehot.scatter_(1,
                                  daily_bike[:, 9, :].long().unsqueeze(1) - 1,
                                  1.0)
    print(daily_weather_onehot)
    daily_bikes = torch.cat((daily_bike, daily_weather_onehot), dim=1)


if __name__ == '__main__':
    main()
