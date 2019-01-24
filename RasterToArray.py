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

# subprocess.call(["gdalbuildvrt", "-resolution", "highest", "-separate", "-dsnodata", "-9999", "stacktest6", #Saved in RasterToArray folder
#     filepath+"MYD04_3K.A2014001.mosaic.061.2018342082532.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
#     filepath+"MYD04_3K.A2014002.mosaic.061.2018342082539.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
#     filepath+"MYD04_3K.A2014003.mosaic.061.2018342082539.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf",
#     filepath+"MYD04_3K.A2014004.mosaic.061.2018342082553.psmcrpgscs_000501286999.Corrected_Optical_Depth_Land_2.hdf"
#     ])

row, col = 200, 160  # Arbitrary point in Luzon

with rasterio.open("stacktest6", 'r') as ds:  # Saved in RasterToArray folder
    arr = ds.read()  # read all raster values

    # Same as arr=ds.read()
    bandlist = list(ds.indexes)
    array = np.zeros((len(bandlist), 611, 360))
    array[:, :, :] = ds.read(bandlist)

    ######################################################################################################
    vals1 = []
    for val in ds.sample([ds.xy(row-1, col-1, offset='ul')]):
        vals1.append(val)

    vals2 = [x for x in ds.sample([ds.xy(row-1, col-1, offset='ul')])]

    # coords = ds.xy(row-1, col-1, offset='ul')
    # vals = ds.sample(coords)

######################################################################################################
print("Array shape:", arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]
print("Source indices:", ds.indexes)
print("Cell upper left:", ds.xy(0, 0, offset='ul'))  # Centroid OR offset to one of ul, ur, ll, lr
print("Source bounds:", ds.bounds)

print("# Rows:", row, "#Cols", col)
print(ds.xy(row, col))

aff = ds.transform
print("Cell width:", aff[0], "Cell height:", -aff[4])
# 0 = width of a pixel
# 1 = row rotation (typically zero)
# 2 = x-coordinate of the upper-left corner of the upper-left pixel
# 3 = column rotation (typically zero)
# 4 = height of a pixel (typically negative)
# 5 = y-coordinate of the of the upper-left corner of the upper-left pixel
######################################################################################################
print("Pixel value:", arr[0, row-1, col-1])  # Band 1 aka Jan 01
print("Pixel value 2:", array[1, row-1, col-1])  # Band 2 aka Jan 02
print(vals1)
print(vals2)
