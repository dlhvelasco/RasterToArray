import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


def setDatetime(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index('datetime')
    df.drop(['Date'], axis=1, inplace=True)
    print(isinstance(df.index, pd.DatetimeIndex))
    return df

csv = pd.read_csv('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/NCR/Pasighour.csv', header=0)
csv = setDatetime(csv)
csv.replace(-9999, np.nan, inplace=True)

csvdaily = csv.resample('D').mean()
csvdaily = csvdaily.replace(np.nan, -9999)
csvdaily.index.names = ['Date']
colnames = csvdaily.columns.values.tolist()
csvdaily.to_csv('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/NCR/Pasig.csv')

# for col in colnames:
#     csvdaily2 = csvdaily[[col]]
#     csvdaily2.columns = ['PM2.5']
#     csvdaily2.to_csv(col+".csv")

# Hourly extraction
# csv1pm = csv[csv.index.hour == 13]

# csv1pmdaily = csv1pm.resample('D').mean()
# csv1pmdaily = csv1pmdaily.replace(np.nan, -9999)
# csv1pmdaily.index.names = ['Date']
# colnames = csv1pmdaily.columns.values.tolist()
# csv1pmdaily.to_csv('/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/NCR/1pm/NCR1518daily_1pm.csv')

# for col in colnames:
#     csv1pmdaily2 = csv1pmdaily[[col]]
#     csv1pmdaily2.columns = ['PM2.5']
#     csv1pmdaily2.to_csv("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/GroundPM/NCR/1pm/"+col+"_1pm.csv")

