import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import geopandas.tools
import shapely.speedups
import rasterio

pd.set_option('display.max_rows', None)
shapely.speedups.enable()


modis_aod = ('MODIS_AOD4326.tif')  # Reference raster grid
ds = rasterio.open(modis_aod, 'r')

# Read the data.
polygons = gpd.GeoDataFrame.from_file('/home/dwight.velasco/dwight.velasco/'
                                      'scratch1/THESIS/RasterToArray/'
                                      'modisgrid/PHGridmap.shp')

# Point data is from VIIRS 02 Jan 2015 to 03 Jan 2015, containing 212 entries
df = pd.read_csv(r'/home/dwight.velasco/dwight.velasco/'
                 'scratch1/THESIS/GroundPM/GroundPM_allsites.txt')
#################################################################
# converting longitude & latitude to geometry
df['coordinates'] = list(zip(df.longitude, df.latitude))
df.coordinates = df.coordinates.apply(Point)
# converting dataframe to geodataframe
points = gpd.GeoDataFrame(df, geometry='coordinates')
points.crs = polygons.crs
sjoin = gpd.tools.sjoin(points, polygons, how='left', op='within')

# converting geodataframe to dataframe
df_sjoin = pd.DataFrame(sjoin)
df_sjoin['monitors'] = 1

counts = df_sjoin.groupby('index_right')['monitors'].sum()
df2 = pd.DataFrame(counts).reset_index()
# map index_right values to coordinates of centroid inside cell
# referencing this coordinate in RasterToArray.py later
df2['coords'] = list(zip(df2['index_right']
                     .map(df_sjoin.drop_duplicates('index_right')
                     .set_index('index_right')['longitude']),
                     df2['index_right']
                     .map(df_sjoin.drop_duplicates('index_right')
                     .set_index('index_right')['latitude'])))
df2['px-py'] = df2['coords'].apply((lambda x: ds.index(x[0], x[-1])))

print(df2)

# len returns number of grids with at least 1 hit for a given day
print("# Unique grids:", len(df2.index))
print("\nTOTAL monitors:", df2.monitors.sum())
