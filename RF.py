import numpy as np
import pandas as pd
from datetime import time
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from subprocess import check_output
from GetTraining import GetTraining
import pickle


def smape(actual, forecast):
    return 100/len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

data = 'rf'

start_time = time.time()
fullDataList = GetTraining(data)
# pickle.dump(fullDataList, open("fullDataList-rf.pkl.dat", "wb"))
# fullDataList = pickle.load(open("fullDataList-rf.pkl.dat", "rb"))

fullData = pd.concat(fullDataList, ignore_index=True)
fullData.eval("wind10m = sqrt(uwind**2 + vwind**2)", engine='numexpr', inplace=True)
# //fullData['wind875'] = np.linalg.norm(fullData[["uwind_875", "vwind_875"]], axis=1)

print("\nTotal valid observations: {} \n".format(len(fullData.index)))
print(fullData.describe())

# plt.figure()
# fullData.plot(subplots=True, figsize=(15, 25))
# plt.title('Scatter Matrix')
# plt.savefig("./plots/scatter.png")

# plt.figure(figsize=(17,14))
# plt.title('Correlation Heatmap')
# cor = fullData.corr(min_periods=3)  #Calculate the correlation of the above variables
# sns.heatmap(cor, square=True, cbar_kws=dict(ticks=np.arange(-1, 1, 0.2)), annot=True, cmap='coolwarm', center=0, annot_kws={"size":9})
# plt.subplots_adjust(left=0.008, right=0.993, top=0.963, bottom=0.236)
# plt.savefig("./plots/heat.png")

labels = np.array(fullData['PM2.5'])
fullData = fullData.drop(['PM2.5', 'omi_no2', 'modis_lst'], axis=1)
# 'merra_pm2.5', 'vwind','uwind','uwind_875','vwind_875', 'wind875', 'wind10m', 'omi_no2', 'modis_lst'

feature_list = list(fullData.columns)
features = np.array(fullData)

# Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth = [5, 10, 30, 70, 100]
max_depth.append(None)

random_grid = {
    # 'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'n_estimators': [600, 1000, 1400, 1800],
    'max_features': [4, 8, 10, int(len(feature_list))],
    'max_depth': max_depth,
    'min_samples_split': [2, 5, 8, 12],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
    }

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 42)

# The baseline predictions are the historical averages
mean_train = np.mean(train_labels)
baseline_predictions = np.ones(train_labels.shape) * mean_train
mae_baseline = metrics.mean_absolute_error(train_labels, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

folds = 5
param_comb = 5

kf = KFold(n_splits=folds, shuffle = True, random_state = 42)

# rf = RandomForestRegressor(n_estimators=1400, max_features='sqrt', max_depth=None, min_samples_split=5, min_samples_leaf=2,bootstrap=False)
rf = RandomForestRegressor(random_state=42, criterion='mae')

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=param_comb, scoring='neg_mean_absolute_error', cv=kf.split(train_features, train_labels), verbose=3, random_state=42, n_jobs=32)

rf_random.fit(train_features, train_labels)
print(rf_random.best_params_)

predictions = rf_random.best_estimator_.predict(test_features)

smape = smape(test_labels, predictions)
print('sMAPE:', round(np.mean(smape), 2), '%.')

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
print('r2_score:', metrics.r2_score(test_labels, predictions))
# print('R^2 .score:', rf_random.best_estimator_.score(test_labels, predictions))

end_time = time.time()
print("\nTime: ", end_time - start_time)

importances = list(rf_random.best_estimator_.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {:4.3f}'.format(*pair)) for pair in feature_importances]

