import rasterio 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from osgeo import gdal
import subprocess

filepath = "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/"
outfile = r"/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/test_{}.hdf"

# ds = gdal.Open(filepath)
# myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(myarray.shape)
# print(myarray)

subprocess.call(["gdalbuildvrt", "-resolution", "highest", "-separate", "stacktest8",  #Saved in RasterToArray folder
    filepath+"MYD04_3K.A2015001.mosaic.061.2019025050128.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
    filepath+"MYD04_3K.A2015002.mosaic.061.2019025050134.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
    filepath+"MYD04_3K.A2015003.mosaic.061.2019025050120.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
    filepath+"MYD04_3K.A2015004.mosaic.061.2019025050135.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
    filepath+"MYD04_3K.A2015005.mosaic.061.2019025050121.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
    filepath+"MYD04_3K.A2015006.mosaic.061.2019025050115.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
    filepath+"MYD04_3K.A2015007.mosaic.061.2019025050123.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf"
    ])

row, col = 200, 160  # Arbitrary point in Luzon

coordinates = (
    (8.95581, 125.59708) , # lon, lat of CSU Ground, Brgy. Ampayon, Butuan City
    (8.95499, 125.52679) , # lon, lat of Butuan City ENR Office Compound, Brgy. Doongan, Butuan City.
)

N = 1

with rasterio.open("stacktest8", 'r') as ds:  # Saved in RasterToArray folder
    arr = ds.read()  # read all raster values

    # Same as arr=ds.read()
    bandlist = list(ds.indexes)
    array = np.zeros((len(bandlist), 610, 359))
    array[:, :, :] = ds.read(bandlist)

    ######################################################################################################
    vals1 = []
    for val in ds.sample([ds.xy(row-1, col-1, offset='ul')]):
        vals1.append(val)

    vals2 = [x for x in ds.sample([ds.xy(row-1, col-1, offset='ul')])]

    ######################################################################################################
    # Obtain row,col given coordinates
    print(ds.index(8.95, 125.59)) # Work on this
    
    for i, (lat, lon) in enumerate(coordinates):
        px, py = ds.index(lat, lon)
        print('Pixel X, Y coords: {}, {}'.format(px, py))
        

        window = rasterio.windows.Window(px, py, N, N)
        print(window)

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = ds.read(window=window)

        # You can then write out a new file
        meta = ds.meta
        meta['width'], meta['height'] = N, N
        meta['transform'] = rasterio.windows.transform(window, ds.transform)

        with rasterio.open(outfile.format(i), 'w', **meta) as dst:
            dst.write(clip)

######################################################################################################
print("\nArray shape:", arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]
print("Source indices:", ds.indexes)
print("Cell upper left:", ds.xy(0, 0, offset='ul'))  # Centroid OR offset to one of ul, ur, ll, lr
print("Source bounds:", ds.bounds)

aff = ds.transform
print("Cell width:", aff[0], "Cell height:", -aff[4])
# 0 = width of a pixel
# 1 = row rotation (typically zero)
# 2 = x-coordinate of the upper-left corner of the upper-left pixel
# 3 = column rotation (typically zero)
# 4 = height of a pixel (typically negative)
# 5 = y-coordinate of the of the upper-left corner of the upper-left pixel
######################################################################################################
print("\nLocation of arbitrary point in Luzon:", ds.xy(row, col))

print("Pixel value:", arr[1, row-1, col-1])  # Band 1 aka Jan 01
print("Pixel value 2:", array[2, row-1, col-1])  # Band 2 aka Jan 02
print(vals1)
print(vals2,"\n")
