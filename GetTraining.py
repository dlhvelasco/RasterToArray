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

# filepath = '/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/'
filepath = '/home/dwight.velasco/scratch1/THESIS/GroundPM/'

coordinates = [
    # (21.12, 114.27)  # upper left (0,0)
    (09.7775000,118.7341667, filepath+'R4B/MIMAROPA_1518.csv'),  
    (08.9549899,125.5267899, filepath+'R13/R13_pms.csv'),  
    (08.9558100,125.5970800, filepath+'R13/R13_doas.csv'),  
    (16.4097200,120.5938900, filepath+'CAR/CAR1518.csv'),
    (13.1658500,123.7516667, filepath+'R5/R51518.csv'),
    (07.1180690,125.6340000, filepath+'R11/r11_doas.csv'),
    (14.4500650,120.9943758,filepath+'NCR/Las_Pinas.csv'),
    (14.5645950,121.0559080, filepath+'NCR/Makati_Pasig.csv'),
    (14.5833967,121.0368167, filepath+'NCR/Mandaluyong.csv'),
    (14.6334111,121.0998333, filepath+'NCR/Marikina.csv'),
    (14.3834944,121.0336750, filepath+'NCR/Muntinlupa.csv'),
    (14.6577167,120.9477000, filepath+'NCR/Navotas.csv'),
    (14.7513425,121.0539344, filepath+'NCR/North_Caloocan.csv'),
    (14.4815269,121.0382578, filepath+'NCR/Paranaque.csv'),
    (14.5286250,121.0081444, filepath+'NCR/Pasay.csv'),
    (14.5467972,121.0667583, filepath+'NCR/Pateros.csv'),
    (14.6055583,121.0322389, filepath+'NCR/San_Juan.csv'),
    (14.5104710,121.0359230, filepath+'NCR/Taguig.csv'),
    (14.6581220,120.9756960, filepath+'NCR/UE_Caloocan.csv'),
    (08.4966667,124.6602778, filepath+'R10/R10_CdeO.csv'),
    (08.5572222,124.5211111, filepath+'R10/R10_ElSalvador.csv'),
    (08.2350556,124.2506111, filepath+'R10/R10_Iligan.csv'),
    (08.5684260,124.7760860, filepath+'R10/R10_Villanueva.csv')
    ]


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
fullData = []

def GetTraining():
    for cy, cx, csvfile in coordinates:

        csv = pd.read_csv(csvfile, header=0)
        csv = setDatetime(csv)

        storedarray = GetPixelValue(dsstore, (cx, cy))
        listedvals = storedarray.listlistedvals
        px = storedarray.px
        py = storedarray.py

        # df_fires = pd.DataFrame(pointspergrid.getFireSpots((px, py)),
        #                         columns=['Date', 'fire_spots'])
        # df_fires = setDatetime(df_fires)

        print('Pixel X, Y coords: {}, {}'.format(px, py))
        training = pd.DataFrame(date_rng, columns=['Date'])
        training['day_of_year'] = date_df['datetime'].dt.dayofyear

        training['modis_aod'] = pd.Series(listedvals[0])
        training['viirs_aod'] = pd.Series(listedvals[21])
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
        training['viirs_dnb'] = ""
        
        training['datetime'] = pd.to_datetime(training['Date'])
        training = training.set_index('datetime')

        storedarrayDate = GetPixelValueDate(dsstoreDate, (cx, cy))
        for idx, dates in enumerate(storedarrayDate.listlisteddates):
            for date in range(len(dates)):
                training.loc[training.index == dates[date], training.columns[idx-len(storedarrayDate.listlisteddates)]] = storedarrayDate.listlistedvals[idx][date]

        training['population'].replace("",np.nan, inplace=True)
        training['modis_evi'].replace("",np.nan, inplace=True)
        training['viirs_dnb'].replace("",np.nan, inplace=True)
        training['population'].fillna(method='ffill', inplace=True)
        training['modis_evi'].fillna(method='ffill', inplace=True)
        training['viirs_dnb'].fillna(method='ffill', inplace=True)
        training.loc[training['viirs_dnb'] < 0, 'viirs_dnb'] = 0
        training.loc[training['modis_evi'] < 0, 'modis_evi'] = 0
        training.loc[training['omi_no2'] <= 0, 'omi_no2'] = np.nan

        # training = pd.merge(training, df_fires, how='left', on=['datetime']) # Fix speed
        training = pd.merge(training, csv,  how='left', on=['datetime'])

        modis_aod_mask = np.isclose(training['modis_aod'], -9.999)
        viirs_aod_mask = np.isclose(training['viirs_aod'], -999)
        modis_lst_mask = np.isclose(training['modis_lst'], 0.0)
        omi_mask = np.isclose(training['omi_no2'], -1.26765e+30)
        ground_mask = np.isclose(training['PM2.5'], -9999)
        training.loc[modis_aod_mask, 'modis_aod'] = np.nan
        training.loc[viirs_aod_mask, 'viirs_aod'] = np.nan
        training.loc[modis_lst_mask, 'modis_lst'] = np.nan
        training.loc[omi_mask, 'omi_no2'] = np.nan
        training.loc[ground_mask, 'PM2.5'] = np.nan
        
        training['AOD'] = training[['modis_aod', 'viirs_aod']].bfill(1).iloc[:,0]
        training.loc[training['AOD'] < 0, 'AOD'] = 0
        training.drop(['Date', 'modis_aod', 'viirs_aod'], axis=1, inplace=True)
        # 'AOD' , 'modis_lst', 'omi_no2'
        # print("Number of valid AOD retrievals", len(training.loc[training['AOD'] >= 0, ['AOD']]))  # count of valid modis retrievals
        # training = training.loc[pd.notnull(training['AOD'])]
        # training = training.loc[pd.notnull(training['modis_lst'])]
        # training = training.loc[pd.notnull(training['omi_no2'])]
        training = training.loc[pd.notnull(training['PM2.5'])]
        print("Number of valid rows", len(training.index))
        fullData.append(training)

        # print(training.describe())

        # plt.figure()
        # training.plot(subplots=True, figsize=(15, 25))
        # plt.title('Scatter Matrix for ' + csvfile[61:-4])
        # plt.savefig("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/plots/" + csvfile[64:-4] + "scatter.png")

    
        # print(training[aodcols].head(21))
        # print(training.dtypes)  # ?population is object for some reason
        # print(training.loc[training.index.year == 2016].head(3))
        # print(training.loc[training.index.year == 2017].head(3))
        # print(training.loc[training.index.year == 2018].head(3))
        # print("Array shape:", training.shape)
        # print(len(training.loc[training['AOD'] >= 0, ['AOD']]))  # count of valid modis retrievals
        # print("Improvement: {:06.4f}%".format(-1*((training['AOD'].isnull().sum()/len(training))*100 - (training['modis_aod'].isnull().sum()/len(training))*100))) # percent missingness
        # print(training['viirs_dnb'].isnull().sum()/len(training)*100)  # percent missingness

    return fullData
        ############################################################################
        # plt.figure(figsize=(17,14))
        # plt.title('Correlation Heatmap for ' + csvfile[61:-4])
        # cor = training.corr(min_periods=3)  #Calculate the correlation of the above variables
        # sns.heatmap(cor, square=True, cbar_kws=dict(ticks=np.arange(-1, 1, 0.2)), annot=True, cmap='coolwarm', center=0, annot_kws={"size":9})
        # # plt.xticks(rotation=30)
        # # plt.tight_layout()
        # plt.subplots_adjust(left=0.008, right=0.993, top=0.963, bottom=0.236)
        # plt.savefig("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/plots/" + csvfile[64:-4] + ".png")
