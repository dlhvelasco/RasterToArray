import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from osgeo import gdal

from pygeotools.lib import iolib, warplib, geolib, timelib, malib


filepath = "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/HDFs/stacktest2"

# ds = gdal.Open(filepath)
# myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(myarray.shape)
# print(myarray)

with rasterio.open(filepath, 'r') as ds:
    arr = ds.read()  # read all raster values

print(arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]