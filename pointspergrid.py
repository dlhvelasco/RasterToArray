import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import geopandas.tools
import shapely.speedups

shapely.speedups.enable()

# Read the data.
polygons = gpd.GeoDataFrame.from_file('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/modisgrid/PHGridmap.shp')

# Point data is from VIIRS 02 Jan 2015 to 03 Jan 2015, containing 212 entries
df = pd.read_csv(r'/home/dwight.velasco/dwight.velasco/scratch1/THESIS/FIRMS/DL_FIRE_V1_39825/fire_archive_V1_39825.csv')
#################################################################
selected_cols = ['latitude', 'longitude', 'acq_date']
df = df[selected_cols]

# converting longitude & latitude to geometry
df['coordinates'] = list(zip(df.longitude, df.latitude))
df.coordinates = df.coordinates.apply(Point)
# converting dataframe to geodataframe
points = gpd.GeoDataFrame(df, geometry='coordinates')
points.crs = polygons.crs
sjoin = gpd.tools.sjoin(points, polygons, how='left')

# converting geodataframe to dataframe
df_sjoin = pd.DataFrame(sjoin)
# add column set to 1 for every hit
df_sjoin['fire_spots'] = 1

cumulative_firespots = 0

# Iterate over dates
grouped = df_sjoin.groupby('acq_date')
for date, group in grouped:
    print("\nDATE:", date)
    # sum all hits
    counts = group.groupby('index_right')['fire_spots'].sum()
    df2 = pd.DataFrame(counts).reset_index()
    # map index_right values to coordinates of centroid inside cell
    # referencing this coordinate in RasterToArray.py later
    df2['coords'] = list(zip(df2['index_right'].map(df_sjoin.drop_duplicates('index_right').set_index('index_right')['longitude']),
                             df2['index_right'].map(df_sjoin.drop_duplicates('index_right').set_index('index_right')['latitude'])))
    print(df2)
    print(len(df2.index))  # len returns number of grids with at least 1 hit for a given day
    print(df2.fire_spots.sum())  # Sum of total fire spots for a given day
    cumulative_firespots += df2.fire_spots.sum()

################################################################## 32651
print("\nTOTAL fire spots:", cumulative_firespots)
