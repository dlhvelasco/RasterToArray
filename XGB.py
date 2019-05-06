import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, gaussian_kde
from scipy.stats.mstats import mquantiles
from datetime import time
from GetTraining import GetTraining
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV, KFold
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib import ticker
from pdpbox import pdp, get_dataset, info_plots
import seaborn as sns; sns.set()
import random
import time
import pickle
import logging
import copy
import sys

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def smape(actual, forecast):
    return 100/len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

data = 'xgb'
# ! USE XGBOOST == 0.81
start_time = time.time()
# fullDataList = GetTraining(data)
# pickle.dump(fullDataList, open("fullDataList-xgb.pkl.dat", "wb"))
fullDataList = pickle.load(open("fullDataList-xgb.pkl.dat", "rb"))

fullData = pd.concat(fullDataList, ignore_index=True)
fullData.eval("wind10m = sqrt(uwind**2 + vwind**2)", engine='numexpr', inplace=True)
fullData.eval("wind875 = sqrt(uwind_875**2 + vwind_875**2)", engine='numexpr', inplace=True)
fullData.eval("frac_vegc = fraction_forest + fraction_vegetation", engine='numexpr', inplace=True)

# fullData[['AOD']] = fullData[['AOD']].fillna(value=-1)
# fullData[['modis_lst']] = fullData[['modis_lst']].fillna(value=-1)

print("AOD Missing %",fullData['AOD'].isnull().sum()/len(fullData)*100)  # percent missingness
print("LST Missing %",fullData['modis_lst'].isnull().sum()/len(fullData)*100)  # percent missingness
print("NO2 Missing %",fullData['omi_no2'].isnull().sum()/len(fullData)*100)  # percent missingness

# print(fullData.describe())
print("\nTotal valid observations: {} \n".format(len(fullData.index)))

fullData = fullData.drop(['uwind_875','vwind_875','wind875', 'r_humidity_875',
'fraction_vegetation', 'fraction_forest', 'fraction_cropland',
], axis=1) # Redundancies

"""
plt.figure(figsize=(17,14))
plt.title('Correlation Heatmap')
cor = fullData.corr(min_periods=3)  #Calculate the correlation of the above variables
sns.heatmap(cor, square=True, cbar_kws=dict(ticks=np.arange(-1, 1, 0.2)), annot=True, cmap='coolwarm', center=0, annot_kws={"size":9})
plt.subplots_adjust(left=0.008, right=0.993, top=0.963, bottom=0.236)
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/heatXGB.png")
sys.exit("Error message")
"""
labels = np.array(fullData['PM2.5'])
fullData = fullData.drop(['PM2.5'], axis=1) 

feature_names_units = ['AOD', 'Day of year', 'Median population', 'VIIRS nighttime lights (nW/cm^2/sr)', 'Eastward wind velocity (m/s)', 'Northward wind velocity (m/s)', '10m wind speed (m/s)', 
'2m dewpoint temperature (K)', '2m air temperature (K)', 'Surface pressure (Pa)', 
'High cloud cover', 'Low cloud cover', 'Total precipitation (m)', 'Evaporation (m)', 'Boundary layer height (m)', 'MODIS enhanced vegetation index', 'Air temperature at 875hPa (K)',
'Surface net solar radiation (J/m^2)', 'Surface net thermal radiation (J/m^2)', 'Specific humidity at 875hPa',
'MERRA-2 PM2.5 concentration (µg/m^3)', 'OMI NO2 concentration (molecules/cm^2)', 'MODIS land surface temperature',
'Fraction of water area', 'Fraction of wetland area', 'Fraction of urban area', 'Fraction of vegetated area']

feature_names = ['AOD', 'Day of year', 'Median population', 'VIIRS nighttime lights', 'Eastward wind velocity', 'Northward wind velocity', '10m wind speed', 
'2m dewpoint temperature', '2m air temperature', 'Surface pressure', 
'High cloud cover', 'Low cloud cover', 'Total precipitation', 'Evaporation', 'Boundary layer height', 'MODIS enhanced vegetation index', 'Air temperature at 875hPa',
'Surface net solar radiation', 'Surface net thermal radiation', 'Specific humidity at 875hPa',
'MERRA-2 PM2.5 concentration', 'OMI NO2 concentration', 'MODIS land surface temperature',
'Fraction of water area', 'Fraction of wetland area', 'Fraction of urban area', 'Fraction of vegetated area']

#! Preserve column orders
feature_abbrv = ['AOD', 'day_of_year', 'population', 'viirs_dnb',  'uwind', 'vwind', 'wind10m', 'dewpt_temp', 'air_temp', 'surface_pressure', 'high_cloud_cover', 'low_cloud_cover', 'total_precipitation', 'evaporation', 'boundary_layer_height', 'modis_evi',  'temp_875',  'surface_net_solar_radiation', 'surface_net_thermal_radiation', 's_humidity_875',
'merra_pm2.5', 'omi_no2', 'modis_lst',
'fraction_water', 'fraction_wetland', 'fraction_urban', 'frac_vegc']

print('Number of features:',len(feature_abbrv))
feature_indices = list(range(0,len(feature_abbrv)+1))

fullData = fullData[feature_abbrv] # Reordering of columns

feature_list = list(fullData.columns)
features = np.array(fullData)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 3057, random_state = 69)

feature_flist = ['f%d'%idx for idx in range(0,len(feature_abbrv))]
df_train_features = pd.DataFrame(train_features, columns=feature_flist)

print(feature_list)
non_nans = (~np.isnan(np.array(test_features))).sum(axis=0)
print(np.where(non_nans == 0, np.nan, non_nans))
# sys.exit("Error message")

mean_train = np.mean(train_labels)
baseline_predictions = np.ones(train_labels.shape) * mean_train
mae_baseline = metrics.mean_absolute_error(train_labels, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

params = {  
    'max_depth': [9,11,13],
    'min_child_weight': [1, 2],
    # 'gamma':[0, 0.01, 0.1, 0.5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'colsample_bynode': [0.6, 0.7, 0.8, 0.9],
    # 'reg_alpha':[0, 0.0005, 0.001, 0.01, 0.1],
    # 'reg_lambda':[0.5, 1, 3, 5, 10],
} 

# feature_monotones = [0]*(len(feature_list))
# monotone_constraints='(' + ','.join([str(m) for m in feature_monotones]) + ')',

xgb = XGBRegressor(random_state=42, n_jobs=8,max_depth=11, min_child_weight=2, gamma=0, subsample=0.9, colsample_bytree=0.9, colsample_bynode=0.9, reg_alpha=0.001, reg_lambda=1, learning_rate=0.05, n_estimators=2500, importance_type="gain")

# kf = KFold(n_splits = 3, shuffle = True, random_state = 42)
# random_search = RandomizedSearchCV(xgb, params, scoring='neg_mean_absolute_error', n_iter=33, random_state=42, cv=kf.split(train_features, train_labels), verbose=20, n_jobs=8, iid=False)
# random_search = GridSearchCV(xgb, params, scoring='neg_mean_absolute_error', cv=kf.split(train_features, train_labels), verbose=12, n_jobs=8, iid=False)

random_search = xgb
# random_search.fit(train_features, train_labels)

# print(random_search.best_params_)
# print(random_search.best_estimator_)
# best_model = random_search.best_estimator_

best_model = random_search
# pickle.dump(best_model, open("model-v74-3057-seed69.dat", "wb"))
best_model = pickle.load(open("model-v74-3057-seed69.dat", "rb"))
print(best_model.get_params(deep=True))
print(best_model.get_xgb_params)


predictions = best_model.predict(test_features) # predictions size determined by test_feature size

smape_A = smape(test_labels, predictions)
print('sMAPE:', round(np.mean(smape_A), 2), '%.')
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
print('r2_score:', metrics.r2_score(test_labels, predictions))

importances = list(best_model.feature_importances_)

# sys.exit("Error message")
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:30} Importance: {:5.4f}'.format(*pair)) for pair in feature_importances]

"""
importances2 = best_model.feature_importances_
indices = np.argsort(importances2)

# Plot the feature importances of the forest
plt.figure(figsize=(14,7))
plt.title("Feature importances")
plt.barh(range(train_features.shape[1]), importances2[indices], color='#6e9be5', align="center")
# If you want to define your own labels, change indices to a list of labels on the following line.
plt.yticks(range(train_features.shape[1]), [feature_names[i] for i in indices])
plt.ylim([-1, train_features.shape[1]])
plt.tight_layout()
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/importances-cover.png")
# sys.exit("Error message")

test_labels = np.array(test_labels)
predictions = np.array(predictions)

xy = np.vstack([test_labels,predictions])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = test_labels[idx], predictions[idx], z[idx]

fig, ax = plt.subplots(figsize=(10,8))
cax = ax.scatter(x, y, c='#6e9be5', s=60, edgecolor='', marker='o', alpha=0.25)
caxb = ax.scatter(x, y, c='#0a49af', s=10, edgecolor='', marker='x', alpha=0.15)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Observed $\mathregular{PM_{2.5}}$ values (µg/$\mathregular{m^3}$)', fontsize=18)
ax.set_ylabel('Predicted $\mathregular{PM_{2.5}}$ values (µg/$\mathregular{m^3}$)', fontsize=18) 
ax.tick_params(axis='both', which='major', labelsize=14)
ax.plot(np.unique(test_labels), np.poly1d(np.polyfit(test_labels, predictions, 1))(np.unique(test_labels)), 'k-')
ax.text(233, 1, 'N = 3057    \n$\mathregular{R^2}$ = 0.87$\,\,\,\,\,\,\,\,\,$ \nMAE = 7.49     \nRMSE = 12.10   \nsMAPE = 29.72%', horizontalalignment='right', fontsize=16)
plt.tight_layout()
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/predictionsXGB.png")


print("Printing dependency plots...")
def partial_dependency(bst, X, y, feature_ids = [], f_id = -1):

    # Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    # 1. Sample a grid of values of a predictor.
    # 2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    
    X_temp = X.copy()
    grid = np.linspace(np.percentile(X_temp[:, f_id], 0), np.percentile(X_temp[:, f_id], 100), 50)
    y_pred = np.zeros(len(grid))

    if len(feature_ids) == 0 or f_id == -1:
        print('Input error!')
        return
    else:
        for i, val in enumerate(grid):
            j = i
            X_temp[:, f_id] = val
            if j >= 2:
                j = j+1
            data = X_temp

            y_pred[i] = np.average(bst.predict(data))

    return grid, y_pred

pdp_names = ['Fraction of forest cover','Population', 'Fraction of vegetated area','Fraction of cropland area', 'Fraction of urban area'] # name of features, not necessary col names
pdp_indices = [26,2,23,24,27] # index of features w.r.t. original df
for idx, f in enumerate(pdp_names):
    print('Printing feature #%d...'%idx)
    plt.close()
    grid, y_pred = partial_dependency(best_model, train_features, test_features, feature_ids = pdp_names, f_id = pdp_indices[idx])

    fig, ax = plt.subplots()
    fig.set_size_inches(7, 5)
    plt.subplots_adjust(left = 0.17, right = 0.94, bottom = 0.15, top = 0.9)

    ax.plot(grid, y_pred, '-', color = '#6e9be5', linewidth = 2.5, label='fit')
    ax.plot(train_features[:, pdp_indices[idx]], train_labels, 'o', color = 'grey', alpha = 0.01)

    deciles = mquantiles(train_features[:, pdp_indices[idx]], prob=np.arange(0.1, 1.0, 0.1))
    trans = mtransforms.blended_transform_factory(ax.transData,ax.transAxes)
    ax.vlines(deciles, [0], 0.05, transform=trans, color='k')

    # prevent x-axis ticks from overlapping
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
    tick_formatter = ScalarFormatter()
    tick_formatter.set_powerlimits((-3, 4))
    ax.xaxis.set_major_formatter(tick_formatter)

    ax.set_xlim(np.nanmin(grid),np.nanmax(grid))
    ax.set_xlabel(f, fontsize = 14)
    ax.set_ylim(0.95 * np.nanmin(y_pred), 1.05 * np.nanmax(y_pred))
    ax.set_ylabel('Partial Dependence', fontsize = 14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc = 'best', fontsize = 12)
    plt.tight_layout()
    fig.set_size_inches(7, 6)
    plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/100-xgb-%d.png"%idx)

print("Printing contour plots...")
# * grid_type: 'equal', 'percentile'

pdp_pop = pdp.pdp_isolate(model=best_model, dataset=df_train_features, model_features=feature_flist, feature='f2', num_grid_points=11, n_jobs=32)
fig, axes = pdp.pdp_plot(pdp_pop, 'Median population', plot_lines=True, frac_to_plot=100, x_quantile=True, plot_pts_dist=True, show_percentile=True)
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/pdp-population.png")
plt.close()

pdp_vegetation = pdp.pdp_isolate(model=best_model, dataset=df_train_features, model_features=feature_flist, feature='f26', num_grid_points=11, n_jobs=32)
fig, axes = pdp.pdp_plot(pdp_vegetation, 'Fraction of vegetated area', plot_lines=True, frac_to_plot=100, x_quantile=True, plot_pts_dist=True, show_percentile=True)
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/pdp-vegetation.png")
plt.close()

pdp_urban = pdp.pdp_isolate(model=best_model, dataset=df_train_features, model_features=feature_flist, feature='f25', num_grid_points=11, n_jobs=32)
fig, axes = pdp.pdp_plot(pdp_urban, 'Fraction of urban area', plot_lines=True, frac_to_plot=100, x_quantile=True, plot_pts_dist=True, show_percentile=True)
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/pdp-urban.png")
plt.close()

pdp_water = pdp.pdp_isolate(model=best_model, dataset=df_train_features, model_features=feature_flist, feature='f23', num_grid_points=11, n_jobs=32)
fig, axes = pdp.pdp_plot(pdp_water, 'Fraction of water area', plot_lines=True, frac_to_plot=100, x_quantile=True, plot_pts_dist=True, show_percentile=True)
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/pdp-water.png")
plt.close()

pdp_wetland = pdp.pdp_isolate(model=best_model, dataset=df_train_features, model_features=feature_flist, feature='f24', num_grid_points=11, n_jobs=32)
fig, axes = pdp.pdp_plot(pdp_wetland, 'Fraction of wetland area', plot_lines=True, frac_to_plot=100, x_quantile=True, plot_pts_dist=True, show_percentile=True)
plt.savefig("/home/dwight.velasco/scratch1/THESIS/Renders/SP-PH/pdp-wetland.png")
plt.close()
"""

end_time = time.time()
print("\nTime: ", end_time - start_time)