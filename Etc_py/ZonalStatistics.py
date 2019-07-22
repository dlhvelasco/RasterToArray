from rasterstats import zonal_stats
import fiona
from osgeo import gdal
import sys
import rasterio
import numpy as np
import collections
from itertools import islice
import pandas as pd

# polygonLayer = '/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Grid/PHGridmap.shp' 
# polygonLayer = '/home/dwight.velasco/scratch1/THESIS/Boundaries/Boundary_Province.shp'
# polygonLayer = '/home/dwight.velasco/scratch1/THESIS/Grid/NCR/NCR_dissolved.shp'
# polygonLayer = '/home/dwight.velasco/scratch1/THESIS/Boundaries/Boundary_Region.shp' 
# polygonLayer = '/home/dwight.velasco/scratch1/THESIS/Boundaries/phl_admbnda_adm0_psa_namria_itos_20180130.shp'
polygonLayer = '/home/dwight.velasco/scratch1/THESIS/Grid/NCR/NCR.shp'

# rasterFilePath = '/home/dwight.velasco/scratch1/THESIS/Renders/PH-raster-v11.3-noelevpaLC-mean.tif'
# rasterFilePath = '/home/dwight.velasco/scratch1/THESIS/Renders/NCR-v83.tif'
# rasterFilePath = '/home/dwight.velasco/scratch1/THESIS/Renders/Limits_calculation/2018b_pop.tif' #total pop
# rasterFilePath = '/home/dwight.velasco/scratch1/THESIS/Renders/Limits_calculation/2018b_safe25.tif'
# rasterFilePath = '/home/dwight.velasco/scratch1/THESIS/Renders/Limits_calculation/PH-raster-v11-noelevpaLC.tif'
rasterFilePath = '/home/dwight.velasco/scratch1/THESIS/Renders/SPP-NCR-raster.tif'

src_ds = gdal.Open(rasterFilePath)
if src_ds is None:
    print('Unable to open INPUT.tif')
    sys.exit(1)

regions = []
nest = []

print("[RASTER BAND COUNT]: ", src_ds.RasterCount)
for band in range(src_ds.RasterCount):
    band += 1
    print("[GETTING BAND]: ", band)
    # srcband = src_ds.GetRasterBand(band)
    # if srcband is None:
    #     continue

    with rasterio.open(rasterFilePath) as src:
        affine = src.transform
        array = src.read(band)
    # print("Calculating zonal stats...")
    stats = zonal_stats(polygonLayer, array, affine=affine, geojson_out=True, all_touched=True, nodata=-9999.0, stats=['min', 'max', 'mean','count','sum','std']) # first 4 cannot be re-ordered
    # *NoDataValue for population: -2147483647; else -9999.0
    # //stats = zonal_stats(polygonLayer, rasterFilePath, band=band, all_touched=True, geojson_out=True)
    # //print(stats[band].keys())  # type id properties geometry
    # //means = [stat['properties'] for stat in stats] # ???
    print([stat['properties'].keys() for stat in stats[:1]]) 
    print([stat['properties'].values() for stat in stats[:1]]) 
    # *[name, min, max, mean, cellcount(centroids?), std] OR
    # *[reg_id, region, prov_id, province, min, max, mean, cellcount(centroids?), std]


    if len(regions) < 1:  # if list is empty
        regions = [next(islice(stat['properties'].values(), 3,4)) for stat in stats] # islice(iterable, stop)
        # MEANS: 2,3 national | 3,4 provincial

    means = [next(islice(stat['properties'].values(), 6,7)) for stat in stats] # islice(iterable, start, stop[, step])
    # MEANS: 12,13 national | 6,7 provincial
    # SUM: 8,9 provincial

    sys.exit("Error message")
    nest.append(means)

df = pd.DataFrame(nest, columns=regions)

# date_rng = pd.date_range(start='1/1/2015', end='12/31/2018')
# df['date'] = date_rng
# df['datetime'] = pd.to_datetime(df['date'])
# df = df.set_index('datetime')
# df.drop(['date'], axis=1, inplace=True)

print(df.head())
# df.to_csv("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/Renders/DailyPM25-NCR.csv", header=True)
# df.to_csv("/home/dwight.velasco/scratch1/THESIS/Renders/AnnualPM25-PH-provincial-v11.3.csv", header=True)
df.to_csv("/home/dwight.velasco/scratch1/THESIS/Renders/Limits_calculation/PopSafe25-Province3.csv", header=True)