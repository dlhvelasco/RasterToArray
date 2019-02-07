import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import geopandas.tools

# Read the data.
polygons = gpd.GeoDataFrame.from_file('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/modisgrid/modisgrid.shp')

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
sjoin = gpd.tools.sjoin(points, polygons, how='inner')

#converting geodataframe to dataframe
df_sjoin = pd.DataFrame(sjoin)
# df_sjoin shape is 213,9

# add obs column set to 1 for every hit
df_sjoin['obs'] = 1

# Iterate over dates
grouped = df_sjoin.groupby('acq_date')
for date, group in grouped:
    print("DATE:", date)
    # sum all hits?
    counts = group.groupby('index_right')['obs'].sum()
    # df should have a total of (154,2) i.e. 154 unique grids(w/ at least 1 hit) between Jan 02 to Jan 03
    df = pd.DataFrame(counts).reset_index()

    print(df.head(), len(df.index)) # len returns number of grids with at least 1 hit for a given day
    print(df.obs.sum()) # Sum of total fire spots for a given day

##################################################################