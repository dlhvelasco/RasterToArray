import rasterio
import numpy as np
import shapely.speedups
import collections

shapely.speedups.enable()

modis_aod =  ('MODIS_AOD4326.vrt', 1, 1)
era5_uwind = ('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/ERA5/era5_data/10m_u_component_of_wind/ERA5_2015_10m_u_component_of_wind4326.vrt', 0.0007222299746216393, 0.5313933787260681)

predictors = [modis_aod, era5_uwind]


def RasterToArray(coordinates):
    listlistedvals = []
    for index, (predictor, scale_factor, offset) in enumerate(predictors):
        ds = rasterio.open(predictor, 'r')  # Saved in RasterToArray folder
        arr = ds.read()  # read all raster values

        # Obtain row,col given coordinates
        (lon, lat) = coordinates
        px, py = ds.index(lon, lat)  # LON = x, LAT =  y
        # print('\nPixel X, Y coords: {}, {}'.format(px, py))

        # Build an NxN window
        window = rasterio.windows.Window(py, px, 1, 1)  # py, px is ul corner; 1 x 1 window

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = ds.read(window=window)
        scaledclip = [(x*scale_factor)+offset for x in np.squeeze(clip).tolist()]  # Int*scale_factor + offset

        listlistedvals.append(scaledclip)
        ListAndIndex = collections.namedtuple('ListAndIndex', 'listlistedvals,index,px,py')(listlistedvals, index, px, py)

    return ListAndIndex

###############################################################################
# aff = ds.transform
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
