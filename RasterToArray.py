import rasterio
import numpy as np
import pandas as pd
import shapely.speedups
import collections

shapely.speedups.enable()

# filename = '/home/dwight.velasco/dwight.velasco/scratch1/THESIS/'
filename = '/home/dwight.velasco/scratch1/THESIS/'
era5files = 'ERA5/era5_data/'

# (filename , scale factor, offset)
modis_aod = (filename + 'MYD04_3K/MODIS_AOD4326.tif', 0.00100000004749745, 0)
viirs_aod = (filename + 'VIIRS/TIF/1518.tif', 1, 0)
modis_lst = (filename + 'MYD11C1/MODIS_LST4326.tif', 0.02, 0)
merra_pm25 = (filename + 'MERRA_AOD/MerraPM25c.tif', 1, 0)
omi_no2 = (filename + 'OMI/OMINO2c.tif', 1, 0)

uwind = (filename + era5files +
		'10m_u_component_of_wind/1518c.tif',
		-0.000982028765824807, -0.1684722900390625)
vwind = (filename + era5files +
		'10m_v_component_of_wind/1518c.tif',
		-0.0009774566242446439, 1.620666503906254)
dewpt_temp = (filename + era5files +
			'2m_dewpoint_temperature/1518c.tif',
			-0.0003536633939334981, 291.74853515625)
air_temp = (filename + era5files +
			'2m_temperature/1518c.tif',
			-0.0004249399983023561, 298.565185546875)
surface_pressure = (filename + era5files +
					'surface_pressure/1518c.tif',
					-0.2986708783495086, 93481.25)
# total_cloud_cover = (filename + era5files +
#                      'total_cloud_cover/1518c.tif',
#                      -0.00001525983668692472, 0.5000038088837755)
high_cloud_cover = (filename + era5files +
					'high_cloud_cover/1518c.tif',
					-0.00001525983618218653, 0.5000037923455238)
low_cloud_cover = (filename + era5files +
					'low_cloud_cover/1518c.tif',
					-0.00001525983601186792, 0.5000037867648643)
surface_net_solar_radiation = (filename + era5files +
								'surface_net_solar_radiation/1518c.tif',
								-56.71749027952828, 1858702.901000977)
surface_net_thermal_radiation = (filename + era5files +
								'surface_net_thermal_radiation/1518c.tif',
								-11.4345615500824, -343551.71875)
total_precipitation = (filename + era5files +
						'total_precipitation/1518c.tif',
						-0.0000006253776231295517, 0.02049112319946289)
evaporation = (filename + era5files +
				'evaporation/1518c.tif',
				-0.00000002996853203093483, -0.0009725653799250722)
boundary_layer_height = (filename + era5files +
						'boundary_layer_height/1518c.tif',
						-0.04024812271908905, 1338.976196289062)

rel_humidity_875 = (filename + era5files +
					'relative_humidity_875hPa/1518c.tif',
					-0.001730378407709606, 54.88664960861206)
spec_humidity_875 = (filename + era5files +
					'specific_humidity_875hPa/1518c.tif',
					-0.0000003084281710785916, 0.010109911496329)
temp_875 = (filename + era5files +
			'temperature_875hPa/1518c.tif',
			-0.0004012233527131783, 286.4755859375)
uwind_875 = (filename + era5files +
			'u_component_of_wind_875hPa/1518c.tif',
			-0.00175793767739425, -0.3765716552734339)
vwind_875 = (filename + era5files +
			'v_component_of_wind_875hPa/1518c.tif',
			-0.001764441965584323, 3.449600219726562)

predictors = [modis_aod, merra_pm25,
				uwind, vwind, dewpt_temp, air_temp, surface_pressure, high_cloud_cover, low_cloud_cover,
				surface_net_solar_radiation, surface_net_thermal_radiation, total_precipitation, evaporation, boundary_layer_height,
				rel_humidity_875, spec_humidity_875, temp_875, uwind_875, vwind_875,
				omi_no2, modis_lst, viirs_aod
]


def RasterToArray():
	print("Obtaining array from raster...")
	listlistedvals = []
	for index, (predictor, scale_factor, offset) in enumerate(predictors):
		print("Processing raster: {}/{}".format(index+1, (len(predictors2)+len(predictors))))
		ds = rasterio.open(predictor, 'r')  # Saved in RasterToArray folder

		listlistedvals.append(ds)

	return listlistedvals


def GetPixelValue(dsstore, coordinates):
	print("\nObtaining pixel values for coordinate...")
	listlistedvals = []
	for index, (predictor, scale_factor, offset) in enumerate(predictors):
		# print("Processing predictor: {}/{}".format(index+1, len(predictors)))
		ds = dsstore[index]

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

# (filename , scale factor, offset, timestep)
population = (filename + 'LandScan/Population.tif', 1, 0, '1/1/2015', 'AS')
modis_evi = (filename + 'MYD13C1/MODIS_EVI4326.tif', 0.0001, 0, '12/27/2014', '16D')
viirs_dnb = (filename + 'VIIRS/DNB/1518.tif', 1, 0, '1/1/2015', 'MS')

predictors2 = [population, modis_evi, viirs_dnb]


def RasterToArrayDate():
	# print("Obtaining array from raster...")
	listlistedvals = []
	for index, (predictor, scale_factor, offset, startdate, timestep) in enumerate(predictors2):
		print("Processing raster: {}/{}".format((index+1+len(predictors)), (len(predictors2)+len(predictors))))
		ds = rasterio.open(predictor, 'r')  # Saved in RasterToArray folder

		listlistedvals.append(ds)

	return listlistedvals

def GetPixelValueDate(dsstoreDate, coordinates):
	print("Obtaining array from raster...")
	listlistedvals = []
	listlisteddates = []
	for index, (predictor, scale_factor, offset, startdate, timestep) in enumerate(predictors2):
		dateslist = pd.date_range(start='1/1/2015', end='12/31/2018', freq=timestep).strftime('%Y-%m-%d').tolist()

		ds = dsstoreDate[index]  # read all raster values

		# Obtain row,col given coordinates
		(lon, lat) = coordinates
		px, py = ds.index(lon, lat)  # LON = x, LAT =  y

		# Build an NxN window
		# py, px is ul corner; 1 x 1 window
		window = rasterio.windows.Window(py, px, 1, 1)

		# Read the data in the window
		# clip is a nbands * N * N numpy array
		clip = ds.read(window=window)
		scaledclip = [(x*scale_factor)+offset for x in np.squeeze(clip).tolist()]

		listlistedvals.append(scaledclip)
		listlisteddates.append(dateslist)

		# print(np.shape(listlistedvals))
		# print(np.shape(listlisteddates))

		ListAndDate = collections.namedtuple('ListAndDate', 'listlistedvals, listlisteddates')(listlistedvals, listlisteddates)

	return ListAndDate

# ////aff = ds.transform
# ////print("Number of dates:", len(ds.indexes))
# /////print("Cell upper left:", ds.xy(0, 0, offset='ul'))  # Centroid OR offset to one of ul, ur, ll, lr
# ////print("Cell upper left:", aff * (0, 0))
# ////print("Source bounds:", ds.bounds)

# ////print("Cell width:", aff[0], "Cell height:", -aff[4])
# ////0 = width of a pixel
# ////1 = row rotation (typically zero)
# ////2 = x-coordinate of the upper-left corner of the upper-left pixel
# ////3 = column rotation (typically zero)
# ////4 = height of a pixel (typically negative)
# ////5 = y-coordinate of the of the upper-left corner of the upper-left pixel

