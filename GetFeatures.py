import numpy as np
import pandas as pd
from datetime import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, GridSearchCV
# from fancyimpute import MICE
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
import csv
from osgeo import gdal
from osgeo import osr
from osgeo import gdal_array
from joblib import Parallel, delayed
import multiprocessing
import itertools
import math

# shapely.speedups.enable()

grid_centroids = '/home/dwight.velasco/scratch1/THESIS/Grid/grid_centroids.csv'
ncr = '/home/dwight.velasco/scratch1/THESIS/Grid/NCR/NCR_centroid.csv'

start_time = time.time()

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
y_up = []
y_lo = []
lats = []
lons = []

data = 'xgb'
best_model = pickle.load(open("model.pkl.dat", "rb"))
# mice = MICE(n_imputations=75, verbose=False)
best_grid = pickle.load(open("y_upper_imp.pkl.dat", "rb"))
best_grid2 = pickle.load(open("y_lower_imp.pkl.dat", "rb"))

"""
with open(ncr) as f:
    header = next(csv.reader(f))
    coordinates=[(float(row[2]), float(row[1])) for row in csv.reader(f)]

print(coordinates[:5])

for cy, cx in coordinates:

    storedarray = GetPixelValue(dsstore, (cx, cy))
    listedvals = storedarray.listlistedvals
    px = storedarray.px
    py = storedarray.py

    print('Pixel X, Y coords: {}, {}'.format(px, py))
    features = pd.DataFrame(date_rng, columns=['Date'])
    features['day_of_year'] = date_df['datetime'].dt.dayofyear

    features['modis_aod'] = pd.Series(listedvals[0])
    features['viirs_aod'] = pd.Series(listedvals[21])
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
    features['r_humidity_875'] = pd.Series(listedvals[14])
    features['s_humidity_875'] = pd.Series(listedvals[15])
    features['temp_875'] = pd.Series(listedvals[16])
    features['uwind_875'] = pd.Series(listedvals[17])
    features['vwind_875'] = pd.Series(listedvals[18])

    features['population'] = ""
    features['modis_evi'] = ""
    features['viirs_dnb'] = ""
    
    features['datetime'] = pd.to_datetime(features['Date'])
    features = features.set_index('datetime')

    storedarrayDate = GetPixelValueDate(dsstoreDate, (cx, cy))
    for idx, dates in enumerate(storedarrayDate.listlisteddates):
        for date in range(len(dates)):
            features.loc[features.index == dates[date], features.columns[idx-len(storedarrayDate.listlisteddates)]] = storedarrayDate.listlistedvals[idx][date]

    features['population'].replace("",np.nan, inplace=True)
    features['modis_evi'].replace("",np.nan, inplace=True)
    features['viirs_dnb'].replace("",np.nan, inplace=True)
    features['population'].fillna(method='ffill', inplace=True)
    features['modis_evi'].fillna(method='ffill', inplace=True)
    features['viirs_dnb'].fillna(method='ffill', inplace=True)
    features.loc[features['viirs_dnb'] < 0, 'viirs_dnb'] = 0
    features.loc[features['modis_evi'] < 0, 'modis_evi'] = 0

    modis_aod_mask = np.isclose(features['modis_aod'], -9.999)
    viirs_aod_mask = np.isclose(features['viirs_aod'], -999)
    modis_lst_mask = np.isclose(features['modis_lst'], 0.0)
    features.loc[modis_aod_mask, 'modis_aod'] = np.nan
    features.loc[viirs_aod_mask, 'viirs_aod'] = np.nan
    features.loc[modis_lst_mask, 'modis_lst'] = np.nan

    features['AOD'] = features[['modis_aod', 'viirs_aod']].bfill(1).iloc[:,0]
    features.loc[features['AOD'] < 0, 'AOD'] = 0
    features.drop(['Date', 'modis_aod', 'viirs_aod'], axis=1, inplace=True)
    
    if data == 'rf':
        features = features.loc[pd.notnull(features['AOD'])]
        features = features.loc[pd.notnull(features['modis_lst'])]
    
    features.eval("wind10m = sqrt(uwind**2 + vwind**2)", engine='numexpr', inplace=True)
    features.eval("wind875 = sqrt(uwind_875**2 + vwind_875**2)", engine='numexpr', inplace=True)
    
    print("Number of valid rows", len(features.index))

    feature_list = list(features.columns)
    features_array = np.array(features)

    print("Predicting features")
    # print(best_model.get_params)

    predictions = best_model.predict(features_array)

    features_array2 = copy.deepcopy(features_array)
    features_array2 = mice.complete(features_array2)
    print("Predicting 95%")
    y_upper = best_grid.predict(features_array2)

    print("Predicting 5%")
    y_lower = best_grid2.predict(features_array2)

    predictions = np.array(predictions)
    y_upper = np.array(y_upper)
    y_lower = np.array(y_lower)

    preds.append(predictions)
    y_up.append(y_upper)
    y_lo.append(y_lower)
    lats.append(cy)
    lons.append(cx)
"""

# pickle.dump(preds, open("preds.pkl", "wb"))
# pickle.dump(y_up, open("y_up.pkl", "wb"))
# pickle.dump(y_lo, open("y_lo.pkl", "wb"))
# pickle.dump(lats, open("lats.pkl", "wb"))
# pickle.dump(lons, open("lons.pkl", "wb"))
preds = pickle.load(open("preds.pkl", "rb"))
y_up = pickle.load(open("y_up.pkl", "rb"))
y_lo = pickle.load(open("y_lo.pkl", "rb"))
lats = pickle.load(open("lats.pkl", "rb"))
lons = pickle.load(open("lons.pkl", "rb"))

# fullFeatures = np.array(list(itertools.chain.from_iterable(preds)))
# print("AOD Missing %",len(fullFeatures[np.where(np.isnan(fullFeatures))]).sum()/len(fullFeatures)*100)  # percent missingness
# print("LST Missing %",fullFeatures['modis_lst'].isnull().sum()/len(fullFeatures)*100)  # percent missingness
# print("\nTotal valid observations: {} \n".format(len(fullFeatures)))
# print(fullFeatures.describe())

preds = np.array(preds)
lats = np.array(lats)
lons = np.array(lons)
print(preds.shape)
print(lats.shape)
print(lons.shape)

############################################################################
# the raster output layer
output_file = 'NCR_raster.tif'
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
xres = 0.0270
yres = 0.0270
nrows = 17
ncols = 9
nbands = 1461
noData = -9999
ncells = 94
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

end_time = time.time()
print("\nTime: ", end_time - start_time)