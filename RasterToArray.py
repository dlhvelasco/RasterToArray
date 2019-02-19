import rasterio
import numpy as np
import subprocess
import shapely.speedups

shapely.speedups.enable()

filepath = r"/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/"
outfile = r"/home/dwight.velasco/dwight.velasco/scratch1/THESIS/MYD04_3K/datacube/test_{}.hdf"

# row, col = 199, 159  # Arbitrary point in Luzon
# Set scale factor, offset for ERA5 data
scale_factor = 0.0007222299746216393
offset = 0.5313933787260681

coordinates = [
    # (21.12, 114.27)  # upper left (0,0)
    # (8.95581, 125.59708),  # lat, lon of CSU Ground, Brgy. Ampayon, Butuan City
    # (8.95499, 125.52679),  # lat, lon of Butuan City ENR Office Compound, Brgy. Doongan, Butuan City.
    # (15.686770491803276, 121.22516713091922)  # Arbitrary pt in Luzon
    (14.524189999999999, 120.60947) # Petron refinery Index #9554
]

# ERA5_2014_10m_u_component_of_wind.nc
# stacktest8.vrt
with rasterio.open("MODIS_AOD4326.vrt", 'r') as ds:  # Saved in RasterToArray folder
    arr = ds.read()  # read all raster values

    # Same as arr=ds.read()
    bandlist = list(ds.indexes)
    # array = np.zeros((len(bandlist), 619, 459))
    array = np.zeros((len(bandlist), arr.shape[1], arr.shape[2]))  # [band, row, col]
    array[:, :, :] = ds.read(bandlist)

    ###########################################################################
    # vals1 = []
    # for val in ds.sample([ds.xy(row, col, offset='ul')]):
    #     vals1.append(val)

    # vals2 = [x for x in ds.sample([ds.xy(row, col, offset='ul')])]

    ###########################################################################
    # Print all bands at a point location
    # Doesnt work? See alt code below
    # for val in ds.sample(coordinates):
    #     print("VAL:", val)

    ###########################################################################
    # Obtain row,col given coordinates
    for i, (lat, lon) in enumerate(coordinates):
        px, py = ds.index(lon, lat)  # LON = x, LAT =  y
        print('\nPixel X, Y coords: {}, {}'.format(px, py))

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

###############################################################################
aff = ds.transform
print("\nArray shape:", arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]
print("Number of dates:", len(ds.indexes))
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

###############################################################################
# print("\nLocation of arbitrary point in Luzon:", ds.xy(row, col))

# print("Pixel value:", arr[1, row, col])  # Band 1 aka Jan 01
# print("Pixel value 2:", array[2, row, col])  # Band 2 aka Jan 02
# print(vals1)
# print(vals2,"\n")
