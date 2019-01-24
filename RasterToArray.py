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

subprocess.call(["gdalbuildvrt", "-resolution", "highest", "-separate", "-srcnodata", "-9999", "stacktest6",                   #Saved in RasterToArray folder
    filepath+"MYD04_3K.A2014001.mosaic.061.2018342082532.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
    filepath+"MYD04_3K.A2014002.mosaic.061.2018342082539.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
    filepath+"MYD04_3K.A2014003.mosaic.061.2018342082539.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
    filepath+"MYD04_3K.A2014004.mosaic.061.2018342082553.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf"
    ])

with rasterio.open("stacktest6", 'r') as ds: #Saved in RasterToArray folder
    arr = ds.read()  # read all raster values

print(arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]
print(arr)
