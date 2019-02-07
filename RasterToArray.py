import rasterio
import numpy as np
# from osgeo import gdal
import subprocess

filepath = "/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/"
outfile = r"/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/test_{}.hdf"

# ds = gdal.Open(filepath)
# myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(myarray.shape)
# print(myarray)

# subprocess.call(["gdalbuildvrt", "-resolution", "highest", "-separate", "stacktest8",  #Saved in RasterToArray folder
#     filepath+"MYD04_3K.A2015001.mosaic.061.2019025050128.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
#     filepath+"MYD04_3K.A2015002.mosaic.061.2019025050134.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
#     filepath+"MYD04_3K.A2015003.mosaic.061.2019025050120.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
#     filepath+"MYD04_3K.A2015004.mosaic.061.2019025050135.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
#     filepath+"MYD04_3K.A2015005.mosaic.061.2019025050121.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
#     filepath+"MYD04_3K.A2015006.mosaic.061.2019025050115.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf",
#     filepath+"MYD04_3K.A2015007.mosaic.061.2019025050123.psmcrpgscs_000501298443.Optical_Depth_Land_And_Ocean.hdf"
#     ])

# row, col = 199, 159  # Arbitrary point in Luzon
scale_factor = 0.0007222299746216393
offset = 0.5313933787260681

coordinates = [
    # (21.066499495936927, 116.90649816843384)  # upper left (0,0)
    # (8.95581, 125.59708),  # lat, lon of CSU Ground, Brgy. Ampayon, Butuan City
    # (8.95499, 125.52679),  # lat, lon of Butuan City ENR Office Compound, Brgy. Doongan, Butuan City.
    (15.686770491803276, 121.22516713091922)  # Arbitrary pt in Luzon
]

# ERA5_2014_10m_u_component_of_wind.nc
# stacktest8
with rasterio.open("ERA5_2014_10m_u_component_of_wind.nc", 'r') as ds:  # Saved in RasterToArray folder
    arr = ds.read()  # read all raster values

    # Same as arr=ds.read()
    bandlist = list(ds.indexes)
    # array = np.zeros((len(bandlist), 610, 359))
    array = np.zeros((len(bandlist), arr.shape[1], arr.shape[2]))  #[band, row, col]
    array[:, :, :] = ds.read(bandlist)

    ######################################################################################################
    # vals1 = []
    # for val in ds.sample([ds.xy(row, col, offset='ul')]):
    #     vals1.append(val)

    # vals2 = [x for x in ds.sample([ds.xy(row, col, offset='ul')])]

    ######################################################################################################
    # Print all bands at a point location
    # Doesnt work? See alt code below
    # for val in ds.sample(coordinates):
    #     print("VAL:", val)

    ######################################################################################################
    # Obtain row,col given coordinates

    for i, (lat, lon) in enumerate(coordinates):
        px, py = ds.index(lon, lat)  # LON = x, LAT =  y
        print('Pixel X, Y coords: {}, {}'.format(px, py))

        # Build an NxN window
        window = rasterio.windows.Window(py, px, 1, 1)  # py, px is ul corner; 1 x 1 window
        # print(window)

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = ds.read(window=window)
        newclip = np.squeeze(clip).tolist()

        # Print all bands at a point location
        print(newclip)

        # scaledclip = [(x*scale_factor)+offset for x in newclip]  # Int*scale_factor + offset
        # print(scaledclip)

        # print([x for x in ds.sample([ds.xy(px, py, offset='ul')])])

        # # You can then write out a new file
        # meta = ds.meta
        # meta['width'], meta['height'] = 1, 1
        # meta['transform'] = rasterio.windows.transform(window, ds.transform)

        # with rasterio.open(outfile.format(i), 'w', **meta) as dst:
        #     dst.write(clip)

######################################################################################################
aff = ds.transform
print("\nArray shape:", arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]
print("Source indices:", ds.indexes)
print("Cell upper left:", ds.xy(0, 0, offset='ul'))  # Centroid OR offset to one of ul, ur, ll, lr
print("Cell upper left:", aff * (0, 0))
print("Source bounds:", ds.bounds)

print("Cell width:", aff[0], "Cell height:", -aff[4])
# 0 = width of a pixel
# 1 = row rotation (typically zero)
# 2 = x-coordinate of the upper-left corner of the upper-left pixel
# 3 = column rotation (typically zero)
# 4 = height of a pixel (typically negative)
# 5 = y-coordinate of the of the upper-left corner of the upper-left pixel

######################################################################################################
# print("\nLocation of arbitrary point in Luzon:", ds.xy(row, col))

# print("Pixel value:", arr[1, row, col])  # Band 1 aka Jan 01
# print("Pixel value 2:", array[2, row, col])  # Band 2 aka Jan 02
# print(vals1)
# print(vals2,"\n")
