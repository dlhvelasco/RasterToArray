import geopandas as gpd
import numpy as np
import itertools
import pandas as pd
import rasterio
from shapely.geometry import Point, shape
import geopandas.tools
import shapely.speedups

pd.set_option('display.max_rows', None)
shapely.speedups.enable()

modis_aod = ('MODIS_AOD4326.tif')  # Reference raster grid
ds = rasterio.open(modis_aod, 'r')

# Read the data.
polygons = gpd.read_file("./modisgrid/PHGridmap.shp")

# /home/dwight.velasco/dwight.velasco/scratch1/THESIS/FIRMS/FIRMS_VIIRS15Day.csv
df = pd.read_csv(r'/home/dwight.velasco/dwight.velasco/'
                 'scratch1/THESIS/FIRMS/FIRMS_VIIRS15Day.csv')
#################################################################
selected_cols = ['latitude', 'longitude', 'acq_date']
df = df[selected_cols]

# converting longitude & latitude to geometry
df['coordinates'] = list(zip(df.longitude, df.latitude))
df.coordinates = df.coordinates.apply(Point)
# converting dataframe to geodataframe
points = gpd.GeoDataFrame(df, geometry='coordinates')
points.crs = polygons.crs
sjoin = gpd.tools.sjoin(points, polygons, how='left', op='within')

# converting geodataframe to dataframe
df_sjoin = pd.DataFrame(sjoin)
# add column set to 1 for every hit
df_sjoin['fire_spots'] = 1

# Iterate over dates
grouped = df_sjoin.groupby('acq_date')


def getFireSpots(pxpy):
    fire_spots_list = []
    for date, group in grouped:
        # sum all hits
        counts = group.groupby('index_right')['fire_spots'].sum()
        df2 = pd.DataFrame(counts).reset_index()

        # map index_right values to coordinates of centroid inside cell
        df2['coords'] = list(zip(df2['index_right']
                                 .map(df_sjoin.drop_duplicates('index_right')
                                 .set_index('index_right')['longitude']),
                                 df2['index_right']
                                 .map(df_sjoin.drop_duplicates('index_right')
                                 .set_index('index_right')['latitude'])))

        df2['px-py'] = df2['coords'].apply((lambda x: ds.index(x[0], x[-1])))

        fire_spots_list.append(df2.loc[df2['px-py'] == pxpy, 'fire_spots']
                                  .values.tolist())

    fire_spots_list = [None if not x else (date, x[0]) for x in fire_spots_list]

    return list(filter(None, fire_spots_list))
