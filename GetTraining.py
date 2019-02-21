import pandas as pd
from datetime import datetime
import numpy as np

import RasterToArray
# import GroundPM_allsites


csv = pd.read_csv('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/R4B/MIMAROPA_15.csv', header=0)
csv['datetime'] = pd.to_datetime(csv['Date'])
csv = csv.set_index('datetime')
csv.drop(['Date'], axis=1, inplace=True)

coordinates = [
    # (21.12, 114.27)  # upper left (0,0)
    (118.73416670000002, 9.7775), # Puerto Princesa
    # (120.60947, 14.524189999999999) # Petron refinery Index #9554
]

date_rng = pd.date_range(start='1/1/2015', end='12/31/2018')

for coords in coordinates:

    listedvals = RasterToArray.RasterToArray(coords).listlistedvals
    print('\nPixel X, Y coords: {}, {}'.format(RasterToArray.RasterToArray(coords).px, RasterToArray.RasterToArray(coords).py))
    training = pd.DataFrame(date_rng, columns=['date'])

    training['MODIS_AOD'] = pd.Series(listedvals[0])
    training['ERA5_UWIND'] = pd.Series(listedvals[1])

    training['datetime'] = pd.to_datetime(training['date'])
    training = training.set_index('datetime')
    training.drop(['date'], axis=1, inplace=True)
    training = pd.merge(training,csv, on=['datetime'])

    print(training.head(10))
