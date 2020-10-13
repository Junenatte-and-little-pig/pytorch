# -*- encoding: utf-8 -*-
import imageio

import torch


def main():
    # ct=pydicom.dcmread('../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083/000000.dcm')
    # print(ct)
    # print(ct.pixel_array)
    # plt.figure(figsize=[10,10])
    # plt.imshow(ct.pixel_array,cmap=plt.bone())
    # plt.show()

    data_dir = '../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083'
    vol_arr = imageio.volread(data_dir, 'DICOM')
    print(vol_arr.shape)
    vol=torch.from_numpy(vol_arr).float()
    vol=torch.unsqueeze(vol,0)
    print(vol.shape)
    


if __name__ == '__main__':
    main()
