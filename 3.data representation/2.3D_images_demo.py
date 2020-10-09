# -*- encoding: utf-8 -*-
import cv2 as cv
import pydicom
import os
import matplotlib.pyplot as plt


def main():
    ct=pydicom.dcmread('../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083/000000.dcm')
    print(ct)
    print(ct.pixel_array)
    plt.figure(figsize=[10,10])
    plt.imshow(ct.pixel_array,cmap=plt.bone())
    plt.show()

    # data_dir='../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083'
    # filenames = [name for name in os.listdir(data_dir) if
    #              os.path.splitext(name)[-1] == '.dcm']
    # for filename in filenames:
    #     ct=pydicom.dcmread(os.path.join(data_dir,filename))
    #     plt.figure(figsize=[10, 10])
    #     plt.imshow(ct.pixel_array, cmap=plt.bone())
    #     plt.show()
    #     plt.cla()
    #     plt.close('all')


if __name__ == '__main__':
    main()
