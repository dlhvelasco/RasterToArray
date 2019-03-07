import geopandas as gpd
import geopandas.tools
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, shape
import shapely.speedups
import pickle

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
shapely.speedups.enable()

modis_aod = ('MODIS_REF_GRID.tif')  # Reference raster grid
ds = rasterio.open(modis_aod, 'r')

try:
        df_sjoin = pd.read_pickle("ppg-df_sjoin.pkl")
        print("Pickle loaded.")
except (OSError, IOError) as e:
        print("No pickle found. Creating pickle...")

        # Read the data.
        polygons = gpd.read_file("./modisgrid/Clean-PHGridmap.shp")

        df = pd.read_csv(r'/home/dwight.velasco/dwight.velasco/'
                         'scratch1/THESIS/FIRMS/FIRMS_VIIRS1518Day.csv')
        #################################################################
        selected_cols = ['latitude', 'longitude', 'acq_date']
        df = df[selected_cols]

        # converting longitude & latitude to geometry
        df['coordinates'] = list(zip(df.longitude, df.latitude))
        df.coordinates = df.coordinates.apply(Point)
        # converting dataframe to geodataframe
        points = gpd.GeoDataFrame(df, geometry='coordinates')
        points.crs = polygons.crs

        poly_mask = polygons.unary_union
        points_clip = points[points.geometry.within(poly_mask)]

        sjoin = gpd.tools.sjoin(points_clip, polygons, how='left', op='within')

        # converting geodataframe to dataframe
        df_sjoin = pd.DataFrame(sjoin)
        # add column set to 1 for every hit
        df_sjoin['fire_spots'] = 1
        df_sjoin.to_pickle("ppg-df_sjoin.pkl", protocol=4)
        print("Pickle created.")

# Iterate over dates
grouped = df_sjoin.groupby('acq_date')


def getFireSpots(pxpy):
    fire_spots_list = []
    print("Obtaining fire spots...")
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
    print("Fire spots obtained.")
    return list(filter(None, fire_spots_list))
