import numpy as np
import pandas as pd
from datetime import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from RasterToArray import RasterToArray, GetPixelValue, RasterToArrayDate, GetPixelValueDate
import time
import pickle
import logging
import copy
import collections
import os
import sys
import csv
from osgeo import gdal
from osgeo import osr
from osgeo import gdal_array
from joblib import Parallel, delayed
import multiprocessing
import itertools
import math
from string import ascii_lowercase

# shapely.speedups.enable()

start_time = time.time()

index = 'xaz'
print(index)

def setDatetime(df):
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index('datetime')
    df.drop(['Date'], axis=1, inplace=True)
    return df

date_rng = pd.date_range(start='1/1/2015', end='12/31/2018')
date_df = pd.DataFrame(date_rng, columns=['Date'])
date_df['datetime'] = pd.to_datetime(date_df['Date'])

dsstore = RasterToArray()
dsstoreDate = RasterToArrayDate()
preds = []
lats = []
lons = []

ncr = '/home/dwight.velasco/scratch1/THESIS/Grid/NCR/NCR_centroid.csv'

with open(ncr) as f:
    header = next(csv.reader(f))
    coordinates=[(float(row[2]), float(row[1])) for row in csv.reader(f)]

lencoords = len(coordinates)
# ! CHANGE MODEL AND OUTPUTNAME
data = 'xgb'
best_model = pickle.load(open("model-v74-3057-seed69.dat", "rb"))
outputfilename = '/home/dwight.velasco/scratch1/THESIS/Renders/NCR-v74-3057-seed69.tif'

landcover_csv = [
"/home/dwight.velasco/scratch1/THESIS/MCD12Q1/115_xy_LC2.csv", 
"/home/dwight.velasco/scratch1/THESIS/MCD12Q1/116_xy_LC2.csv",
"/home/dwight.velasco/scratch1/THESIS/MCD12Q1/117_xy_LC2.csv", 
"/home/dwight.velasco/scratch1/THESIS/MCD12Q1/118_xy_LC2.csv"
] # LC2: fixed veg, +wetland, +cropland

landcover_types = [
# 'fraction_cropland', 
'fraction_forest', 
'fraction_vegetation',
'fraction_wetland',
'fraction_urban',
'fraction_water',
]


for cooordidx, (cy, cx) in enumerate(coordinates):
    print("\nProcessing: %d/%d ..." %(cooordidx+1, lencoords))

    storedarray = GetPixelValue(dsstore, (cx, cy))
    listedvals = storedarray.listlistedvals
    px = storedarray.px
    py = storedarray.py
    print('Pixel X, Y coords: {}, {}'.format(px, py))

    features = pd.DataFrame(date_rng, columns=['Date'])
    features['day_of_year'] = date_df['datetime'].dt.dayofyear

    features['modis_aod'] = pd.Series(listedvals[0])
    features['viirs_aod'] = pd.Series(listedvals[21])

    features['merra_pm2.5'] = pd.Series(listedvals[1])
    features['merra_pm2.5'] *= 1000000000
    features['omi_no2'] = pd.Series(listedvals[19])
    features['modis_lst'] = pd.Series(listedvals[20])

    features['uwind'] = pd.Series(listedvals[2])
    features['vwind'] = pd.Series(listedvals[3])
    features['dewpt_temp'] = pd.Series(listedvals[4])
    features['air_temp'] = pd.Series(listedvals[5])
    features['surface_pressure'] = pd.Series(listedvals[6])
    features['high_cloud_cover'] = pd.Series(listedvals[7])
    features['low_cloud_cover'] = pd.Series(listedvals[8])
    features['surface_net_solar_radiation'] = pd.Series(listedvals[9])
    features['surface_net_thermal_radiation'] = pd.Series(listedvals[10])
    features['total_precipitation'] = pd.Series(listedvals[11])
    features['evaporation'] = pd.Series(listedvals[12])
    features['boundary_layer_height'] = pd.Series(listedvals[13])
    # //features['r_humidity_875'] = pd.Series(listedvals[14])
    features['s_humidity_875'] = pd.Series(listedvals[15])
    features['temp_875'] = pd.Series(listedvals[16])
    # //features['uwind_875'] = pd.Series(listedvals[17])
    # //features['vwind_875'] = pd.Series(listedvals[18])

    features['population'] = ""
    features['modis_evi'] = ""
    features['viirs_dnb'] = ""
    
    features['datetime'] = pd.to_datetime(features['Date'])
    features = features.set_index('datetime')

    storedarrayDate = GetPixelValueDate(dsstoreDate, (cx, cy))
    for idx, dates in enumerate(storedarrayDate.listlisteddates):
        for date in range(len(dates)):
            features.loc[features.index == dates[date], features.columns[idx-len(storedarrayDate.listlisteddates)]] = storedarrayDate.listlistedvals[idx][date]

    features['fraction_forest'] = ""
    features['fraction_vegetation'] = ""
    features['fraction_wetland'] = ""
    # features['fraction_cropland'] = ""
    features['fraction_urban'] = ""
    features['fraction_water'] = ""
    # ! SET PROPERLY
    for landcover_type in landcover_types:
        for idx, year in enumerate(range(2015,2019)):
            df = pd.read_csv(landcover_csv[idx], index_col=0)
            features.loc[features.index == '%d-01-01'%year, landcover_type] = float(df.loc[df['px-py'] == '(%d, %d)' %(px,py), landcover_type])

    features['fraction_forest'].replace("",np.nan, inplace=True)
    features['fraction_vegetation'].replace("",np.nan, inplace=True)
    features['fraction_wetland'].replace("",np.nan, inplace=True)
    # features['fraction_cropland'].replace("",np.nan, inplace=True)
    features['fraction_urban'].replace("",np.nan, inplace=True)
    features['fraction_water'].replace("",np.nan, inplace=True)

    features['fraction_forest'].fillna(method='ffill', inplace=True)
    features['fraction_vegetation'].fillna(method='ffill', inplace=True)
    features['fraction_wetland'].fillna(method='ffill', inplace=True)
    # features['fraction_cropland'].fillna(method='ffill', inplace=True)
    features['fraction_urban'].fillna(method='ffill', inplace=True)
    features['fraction_water'].fillna(method='ffill', inplace=True)

    features['population'].replace("",np.nan, inplace=True)
    features['modis_evi'].replace("",np.nan, inplace=True)
    features['viirs_dnb'].replace("",np.nan, inplace=True)
    features['population'].fillna(method='ffill', inplace=True)
    features['modis_evi'].fillna(method='ffill', inplace=True)
    features['viirs_dnb'].fillna(method='ffill', inplace=True)
    features.loc[features['viirs_dnb'] < 0, 'viirs_dnb'] = 0
    features.loc[features['modis_evi'] < 0, 'modis_evi'] = 0
    features.loc[features['omi_no2'] <= 0, 'omi_no2'] = np.nan

    modis_aod_mask = np.isclose(features['modis_aod'], -9.999)
    viirs_aod_mask = np.isclose(features['viirs_aod'], -999)
    modis_lst_mask = np.isclose(features['modis_lst'], 0.0)
    omi_mask = np.isclose(features['omi_no2'], -1.26765e+30)

    features.loc[modis_aod_mask, 'modis_aod'] = np.nan
    features.loc[viirs_aod_mask, 'viirs_aod'] = np.nan
    features.loc[modis_lst_mask, 'modis_lst'] = np.nan
    features.loc[omi_mask, 'omi_no2'] = np.nan

    features['AOD'] = features[['modis_aod', 'viirs_aod']].bfill(1).iloc[:,0]
    features.loc[features['AOD'] < 0, 'AOD'] = 0
    features.drop(['Date', 'modis_aod', 'viirs_aod'], axis=1, inplace=True)
    
    if data == 'rf':
        features = features.loc[pd.notnull(features['AOD'])]
    
    features.eval("wind10m = sqrt(uwind**2 + vwind**2)", engine='numexpr', inplace=True)
    # //features.eval("wind875 = sqrt(uwind_875**2 + vwind_875**2)", engine='numexpr', inplace=True)
    
    features.eval("frac_vegc = fraction_forest + fraction_vegetation", engine='numexpr', inplace=True)
    features = features.drop(['fraction_forest','fraction_vegetation'], axis=1) 
    
    # print("Number of valid rows", len(features.index))
    print("Number of columns", len(features.columns))
    
    #! Preserve column orders
    feature_abbrv = ['AOD', 'day_of_year', 'population', 'viirs_dnb',  'uwind', 'vwind', 'wind10m', 'dewpt_temp', 'air_temp', 'surface_pressure', 'high_cloud_cover', 'low_cloud_cover', 'total_precipitation', 'evaporation', 'boundary_layer_height', 'modis_evi',  'temp_875',  'surface_net_solar_radiation', 'surface_net_thermal_radiation', 's_humidity_875',
    'merra_pm2.5', 'omi_no2', 'modis_lst',
    'fraction_water', 'fraction_wetland', 'fraction_urban', 'frac_vegc']
    
    features = features[feature_abbrv]
    # print(features.columns)

    features_array = np.array(features)
    lats.append(cy)
    lons.append(cx)

    print("Predicting features")
    predictions = best_model.predict(features_array)
    predictions = np.array(predictions)
    preds.append(predictions)


# *preds, lats, lons are lists

# preds = list(itertools.chain.from_iterable(preds))
# lats = list(itertools.chain.from_iterable(lats))
# lons = list(itertools.chain.from_iterable(lons))

# pickle.dump(preds, open("/home/dwight.velasco/scratch1/THESIS/Grid/predsFull.pkl", "wb"))
# preds = pickle.load(open("/home/dwight.velasco/scratch1/THESIS/Grid/predsFull.pkl", "rb"))
# lats = pickle.load(open("/home/dwight.velasco/scratch1/THESIS/Grid/latsFull.pkl", "rb"))
# lons = pickle.load(open("/home/dwight.velasco/scratch1/THESIS/Grid/lonsFull.pkl", "rb"))

# //fullFeatures = np.array(list(itertools.chain.from_iterable(preds)))
# //print("AOD Missing %",len(fullFeatures[np.where(np.isnan(fullFeatures))]).sum()/len(fullFeatures)*100)  # percent missingness

preds = np.array(preds)
lats = np.array(lats)
lons = np.array(lons)

print(preds.shape)
print(lats.shape)
print(lons.shape)

############################################################################
# the raster output layer
output_file = outputfilename
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
xres = 0.0270
yres = 0.0270
nrows = 17 # NCR 17 PH 601
ncols = 10 # NCR 10 PH 359
nbands = 1461
noData = -9999
array = np.empty((nrows, ncols), dtype=np.float32)

xmin,ymin,xmax,ymax = [lons.min()-xres/2, lats.min()-yres/2,lons.max()+xres/2, lats.max()+yres/2]
geotransform=(xmin, xres, 0, ymax, 0, -yres)

output_raster = gdal.GetDriverByName('GTiff').Create(output_file, ncols, nrows, nbands, gdal.GDT_Float32)  # Open the file
output_raster.SetGeoTransform(geotransform)
output_raster.SetProjection(srs.ExportToWkt())

# Loop bands
for i in range(nbands):
    # Init array with nodata
    array[:] = noData
    # Loop lat/lons inc. index j
    for j, (lon, lat) in enumerate(zip(lons, lats)):
        # Calc x, y pixel index
        x = math.floor((lon - xmin) / xres)
        y = math.floor((lat - ymin) / yres)

        # Fill the array at y, x with the value from predictions at band i, index j 
        array[y, x] = preds[j, i]
        
    array = np.flipud(array)
    output_raster.GetRasterBand(i+1).WriteArray(array)
    output_raster.GetRasterBand(i+1).SetNoDataValue(noData)

del output_raster

print(index)
end_time = time.time()
print("\nTime: ", end_time - start_time)