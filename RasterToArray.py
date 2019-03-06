import rasterio
import numpy as np
import shapely.speedups
import collections

shapely.speedups.enable()

filename = '/home/dwight.velasco/dwight.velasco/scratch1/THESIS/'
era5files = 'ERA5/era5_data/'

# (filename , scale factor, offset)
modis_aod = ('MODIS_AOD4326.tif', 1, 0)

uwind = (filename + era5files
                + '10m_u_component_of_wind/1518.tif',
                -0.000982028765824807, -0.1684722900390625)      
vwind = (filename + era5files
                + '10m_v_component_of_wind/1518.tif',
                -0.0009774566242446439, 1.620666503906254)
2m_dew_temp = (filename + era5files
                + '2m_dewpoint_temperature/1518.tif',
                -0.0003536633939334981, 291.74853515625)
surface_pressure = (filename + era5files
                + 'surface_pressure/1518.tif',
                -0.2986708783495086, 93481.25)
# total_cloud_cover = (filename + era5files
#                 + 'total_cloud_cover/1518.tif',
#                 -0.00001525983668692472, 0.5000038088837755)
high_cloud_cover = (filename + era5files
                + 'high_cloud_cover/1518.tif',
                -0.00001525983618218653, 0.5000037923455238)
low_cloud_cover = (filename + era5files
                + 'low_cloud_cover/1518.tif',
                -0.00001525983601186792, 0.5000037867648643)
total_precipitation = (filename + era5files
                + 'total_precipitation/1518.tif',
                -0.0000006253776231295517, 0.02049112319946289)
evaporation = (filename + era5files
                + 'evaporation/1518.tif',
                -0.00000002996853203093483, -0.0009725653799250722)
boundary_layer_height = (filename + era5files
                + 'boundary_layer_height/1518.tif',
                -0.04024812271908905, 1338.976196289062)


merra_pm25 = ('MerraPM25.tif', 1, 0)

predictors = [modis_aod, merra_pm25,
              uwind, vwind, 2m_dew_temp, surface_pressure, high_cloud_cover, low_cloud_cover, total_precipitation, evaporation, boundary_layer_height
             ]


def RasterToArray(coordinates):
    print("Obtaining array from raster...")
    listlistedvals = []
    for index, (predictor, scale_factor, offset) in enumerate(predictors):
        print("Processing array: {}/{}".format(index,len(predictors)))
        ds = rasterio.open(predictor, 'r')  # Saved in RasterToArray folder
        arr = ds.read()  # read all raster values

        # Obtain row,col given coordinates
        (lon, lat) = coordinates
        px, py = ds.index(lon, lat)  # LON = x, LAT =  y

        # Build an NxN window
        # py, px is ul corner; 1 x 1 window
        window = rasterio.windows.Window(py, px, 1, 1)

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = ds.read(window=window)

        # Int*scale_factor + offset
        scaledclip = [(x*scale_factor)+offset for x in np.squeeze(clip).tolist()]

        listlistedvals.append(scaledclip)
        ListAndIndex = collections.namedtuple('ListAndIndex', 'listlistedvals, index, px, py')(listlistedvals, index, px, py)
    
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
