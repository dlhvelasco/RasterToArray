import rasterio
import shapely.speedups
import collections

shapely.speedups.enable()

# Set scale factor, offset for ERA5 data
scale_factor = 0.0007222299746216393
offset = 0.5313933787260681

def RasterToArray(coordinates):
    print(coordinates)
    with rasterio.open("MODIS_AOD4326.vrt", 'r') as ds:  # Saved in RasterToArray folder
        arr = ds.read()  # read all raster values
        # # Same as arr=ds.read()
        # bandlist = list(ds.indexes)
        # # array = np.zeros((len(bandlist), 619, 459))
        # array = np.zeros((len(bandlist), arr.shape[1], arr.shape[2]))  # [band, row, col]
        # array[:, :, :] = ds.read(bandlist)
        ###########################################################################
        # Print all bands at a point location
        # Takes list/iterator of LON,LAT > outputs ndarray
        # for val in ds.sample(coordinates): # coordinates aka coords is not a iterable if we want to pass one coord at a time
        #     listedvals =  val.tolist()

    ###########################################################################
    # Obtain row,col given coordinates
    for i, (lon, lat) in enumerate(coordinates):
        px, py = ds.index(lon, lat)  # LON = x, LAT =  y
        # print('\nPixel X, Y coords: {}, {}'.format(px, py))
        
        # Build an NxN window
        window = rasterio.windows.Window(py, px, 1, 1)  # py, px is ul corner; 1 x 1 window
        # print(window)

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = ds.read(window=window)
        listedvals = np.squeeze(clip).tolist()

        ListAndIndex = collections.namedtuple('ListAndIndex','listedvals,px,py')(listedvals,px,py)

        # # scaledclip = [(x*scale_factor)+offset for x in newclip]  # Int*scale_factor + offset
        # # print(scaledclip)

        # # Print all bands at a point location
        # print(listedvals)

        # # You can then write out a new file
        # meta = ds.meta
        # meta['width'], meta['height'] = 1, 1
        # meta['transform'] = rasterio.windows.transform(window, ds.transform)

        # with rasterio.open(outfile.format(i), 'w', **meta) as dst:
        #     dst.write(clip)

        ###########################################################################
        # Does the same as above but prints dtype
        # vals1 = []
        # for val in ds.sample([ds.xy(px, py, offset='ul')]):
        #     vals1.append(val)

        # vals2 = [x for x in ds.sample([ds.xy(px, py, offset='ul')])]
    return ListAndIndex

###############################################################################
# aff = ds.transform
# print("\nArray shape:", arr.shape)  # this is a 3D numpy array, with dimensions [band, row, col]
# print("Number of dates:", len(ds.indexes))
# print("Cell upper left:", ds.xy(0, 0, offset='ul'))  # Centroid OR offset to one of ul, ur, ll, lr
# print("Cell upper left:", aff * (0, 0))
# print("Source bounds:", ds.bounds)

# print("Cell width:", aff[0], "Cell height:", -aff[4])
# 0 = width of a pixel
# 1 = row rotation (typically zero)
# 2 = x-coordinate of the upper-left corner of the upper-left pixel
# 3 = column rotation (typically zero)
# 4 = height of a pixel (typically negative)
# 5 = y-coordinate of the of the upper-left corner of the upper-left pixel

###############################################################################
