import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from pandas.plotting import scatter_matrix

# import GroundPM_allsites
from RasterToArray import RasterToArray, GetPixelValue, RasterToArrayDate, GetPixelValueDate
import pointspergrid

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

filepath = '/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/'

coordinates = [
    # (21.12, 114.27)  # upper left (0,0)
    (118.73416670000002, 9.7775, filepath+'R4B/MIMAROPA_1518.csv'),  # Puerto Princesa (411, 67)
    # (125.52678999999999, 8.954989999999999, filepath+'R13/R13_pms.csv'),  # R13_BUTUAN_PMS (441, 318)
    # (125.59708, 8.955810000000001, filepath+'R13/R13_doas.csv'),  # R13_BUTUAN_DOAS (441, 321)
    # (120.59389000000000181, 16.40972000000000008, filepath+'CAR/CAR1518.csv')
    ]


def setDatetime(df):
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index('datetime')
    df.drop(['Date'], axis=1, inplace=True)
    return df

date_rng = pd.date_range(start='1/1/2015', end='12/31/2018')

dsstore = RasterToArray()
dsstoreDate = RasterToArrayDate()

for cx, cy, csvfile in coordinates:

    csv = pd.read_csv(csvfile, header=0)
    csv = setDatetime(csv)

    storedarray = GetPixelValue(dsstore, (cx, cy))
    listedvals = storedarray.listlistedvals
    px = storedarray.px
    py = storedarray.py

    # df_fires = pd.DataFrame(pointspergrid.getFireSpots((px, py)),
    #                         columns=['Date', 'fire_spots'])
    # df_fires = setDatetime(df_fires)

    print('\nPixel X, Y coords: {}, {}'.format(px, py))
    training = pd.DataFrame(date_rng, columns=['Date'])

    training['modis_aod'] = pd.Series(listedvals[0])
    training['merra_pm2.5'] = pd.Series(listedvals[1])
    training['merra_pm2.5'] *= 1000000000
    training['omi_no2'] = pd.Series(listedvals[19])
    training['modis_lst'] = pd.Series(listedvals[20])

    training['uwind'] = pd.Series(listedvals[2])
    training['vwind'] = pd.Series(listedvals[3])
    training['dewpt_temp'] = pd.Series(listedvals[4])
    training['air_temp'] = pd.Series(listedvals[5])
    training['surface_pressure'] = pd.Series(listedvals[6])
    training['high_cloud_cover'] = pd.Series(listedvals[7])
    training['low_cloud_cover'] = pd.Series(listedvals[8])
    training['surface_net_solar_radiation'] = pd.Series(listedvals[9])
    training['surface_net_thermal_radiation'] = pd.Series(listedvals[10])
    training['total_precipitation'] = pd.Series(listedvals[11])
    training['evaporation'] = pd.Series(listedvals[12])
    training['boundary_layer_height'] = pd.Series(listedvals[13])
    training['r_humidity_875'] = pd.Series(listedvals[14])
    training['s_humididity_875'] = pd.Series(listedvals[15])
    training['temp_875'] = pd.Series(listedvals[16])
    training['uwind_875'] = pd.Series(listedvals[17])
    training['vwind_875'] = pd.Series(listedvals[18])

    training['population'] = ""
    training['modis_evi'] = ""

    training = setDatetime(training)
    # -2,0 -1,1
    # -3,0 -2,1, -1,2

    storedarrayDate = GetPixelValueDate(dsstoreDate, (cx, cy))
    for idx, dates in enumerate(storedarrayDate.listlisteddates):
        for date in range(len(dates)):
            training.loc[training.index == dates[date], training.columns[idx-len(storedarrayDate.listlisteddates)]] = storedarrayDate.listlistedvals[idx][date]

    training['population'].replace("",np.nan, inplace=True)
    training['modis_evi'].replace("",np.nan, inplace=True)
    training['population'].fillna(method='ffill', inplace=True)
    training['modis_evi'].fillna(method='ffill', inplace=True)

    # training = pd.merge(training, df_fires, how='left', on=['datetime']) # Fix speed
    training = pd.merge(training, csv,  how='left', on=['datetime'])

    modis_aod_mask = np.isclose(training['modis_aod'], -9.999)
    modis_lst_mask = np.isclose(training['modis_lst'], 0.0)
    omi_mask = np.isclose(training['omi_no2'], -1.267651e+30)
    ground_mask = np.isclose(training['PM2.5'], -9999)
    training.loc[modis_aod_mask, 'modis_aod'] = np.nan
    training.loc[modis_lst_mask, 'modis_lst'] = np.nan
    training.loc[omi_mask, 'omi_no2'] = np.nan
    training.loc[ground_mask, 'PM2.5'] = np.nan

    # print(training.head(2))
    print(training.tail(2))
    # print(training.dtypes)  # population is object for some reason
    # print(training.loc[training.index.year == 2016].head(3))
    # print(training.loc[training.index.year == 2017].head(3))
    # print(training.loc[training.index.year == 2018].head(3))
    # print("Array shape:", training.shape)

    # print(len(training.loc[training['modis_aod'] >= 0, ['modis_aod']]))  # count of valid modis retrievals
    # print(training.isnull().sum()/len(training)*100)  # percent missingness


    ############################################################################
    # plt.figure(figsize=(16,13))
    # plt.title('Correlation Heatmap for ' + csvfile[61:-4])
    # cor = training.corr(min_periods=3)  #Calculate the correlation of the above variables
    # sns.heatmap(cor, square=True, cbar_kws=dict(ticks=np.arange(-1, 1, 0.2)), annot=True, cmap='coolwarm', center=0, annot_kws={"size":9})
    # # plt.xticks(rotation=30)
    # # plt.tight_layout()
    # plt.subplots_adjust(left=0.008, right=0.993, top=0.963, bottom=0.236)
    # plt.savefig("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/plots/" + csvfile[64:-4] + ".png")

    # plt.figure()
    # training.plot(subplots=True, figsize=(13, 20))
    # plt.title('Scatter Matrix for ' + csvfile[61:-4])
    # plt.savefig("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/plots/" + csvfile[64:-4] + "scatter.png")
