import pandas as pd
from datetime import datetime
import numpy as np

# import GroundPM_allsites
from RasterToArray import RasterToArray
import pointspergrid

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

filepath = '/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/'

coordinates = [
    # (21.12, 114.27)  # upper left (0,0)
    (118.73416670000002, 9.7775, filepath+'R4B/MIMAROPA_15.csv'),  # Puerto Princesa (411, 67)
    (125.52678999999999, 8.954989999999999, filepath+'R13/R13_pms.csv'),  # R13_BUTUAN_PMS (441, 318)
    # (125.59708, 8.955810000000001),  # R13_BUTUAN_DOAS (441, 321)
    # (121.09231940000001, 14.6354333) # MARIKINA_ADDITION (231, 154)
    # (121.0998333,14.6334111) # MANDALUYONG_CITYHALL (231, 154)
]


def setDatetime(df):
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index('datetime')
    df.drop(['Date'], axis=1, inplace=True)
    return df

date_rng = pd.date_range(start='1/1/2015', end='12/31/2018')

for cx, cy, csvfile in coordinates:

    csv = pd.read_csv(csvfile, header=0)
    csv = setDatetime(csv)

    storedarray = RasterToArray((cx, cy))
    listedvals = storedarray.listlistedvals
    px = storedarray.px
    py = storedarray.py

    df_fires = pd.DataFrame(pointspergrid.getFireSpots((px, py)),
                            columns=['Date', 'fire_spots'])
    df_fires = setDatetime(df_fires)

    print('\nPixel X, Y coords: {}, {}'.format(px, py))
    training = pd.DataFrame(date_rng, columns=['Date'])

    training['modis_aod'] = pd.Series(listedvals[0])
    training['merra_pm2.5'] = pd.Series(listedvals[1])
    training['merra_pm2.5'] *= 1000000000

    training['uwind'] = pd.Series(listedvals[2])
    training['vwind'] = pd.Series(listedvals[3])
    training['2m_dew_temp'] = pd.Series(listedvals[4])
    training['surface_pressure'] = pd.Series(listedvals[5])
    training['total_cloud_cover'] = pd.Series(listedvals[6])

    training = setDatetime(training)

    training = pd.merge(training, csv,  how='left', on=['datetime'])
    training = pd.merge(training, df_fires, how='left', on=['datetime'])

    print(training.head(7))
    print("Array shape:", training.shape)
