import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from osgeo import gdal
import subprocess

filepath = "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/HDFs/stacktests/"

# ds = gdal.Open(filepath)
# myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(myarray.shape)
# print(myarray)

# subprocess.call(["gdalbuildvrt", "-resolution", "highest", "-separate", "-srcnodata", "-9999", "stacktest6", #Saved in RasterToArray folder
#     filepath+"MYD04_3K.A2014001.mosaic.061.2018342082532.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
#     filepath+"MYD04_3K.A2014002.mosaic.061.2018342082539.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
#     filepath+"MYD04_3K.A2014003.mosaic.061.2018342082539.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
#     filepath+"MYD04_3K.A2014004.mosaic.061.2018342082553.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf"
#     ])

with rasterio.open("stacktest6", 'r') as ds:  # Saved in RasterToArray folder
    arr = ds.read()  # read all raster values

    # Same as arr=ds.read()
    bandlist = list(ds.indexes)
    array = np.zeros((len(bandlist), 611, 360))
    array[:, :, :] = ds.read(bandlist)
    gt = ds.transform

print("Array shape:", arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]
print("Source indices:", ds.indexes)
print("Cell upper left:", ds.xy(0, 0, offset='ul'))  # Centroid offset to one of ul, ur, ll, lr
print("Source bounds:", ds.bounds)
row, col = ds.index(ds.bounds.right, ds.bounds.bottom)
print(row, col)

print(arr[1, row-1, col-1])
print(array[1, row-1, col-1])

print(gt)
# a = width of a pixel
# b = row rotation (typically zero)
# c = x-coordinate of the upper-left corner of the upper-left pixel
# d = column rotation (typically zero)
# e = height of a pixel (typically negative)
# f = y-coordinate of the of the upper-left corner of the upper-left pixel
print(gt[0], -gt[4])
