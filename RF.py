import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from subprocess import check_output
from datetime import time

from GetTraining import GetTraining


fullDataList = GetTraining()
fullData = pd.concat(fullDataList, ignore_index=True)
print("\nTotal valid observations: {} \n".format(len(fullData.index)))
print(fullData.describe())

# plt.figure()
# fullData.plot(subplots=True, figsize=(15, 25))
# plt.title('Scatter Matrix')
# plt.savefig("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/plots/scatter.png")

# plt.figure(figsize=(17,14))
# plt.title('Correlation Heatmap')
# cor = fullData.corr(min_periods=3)  #Calculate the correlation of the above variables
# sns.heatmap(cor, square=True, cbar_kws=dict(ticks=np.arange(-1, 1, 0.2)), annot=True, cmap='coolwarm', center=0, annot_kws={"size":9})
# plt.subplots_adjust(left=0.008, right=0.993, top=0.963, bottom=0.236)
# plt.savefig("/home/dwight.velasco/dwight.velasco/scratch1/THESIS/RasterToArray/plots/heat.png")


labels = np.array(fullData['PM2.5'])
fullData = fullData.drop(['PM2.5'], axis=1)

feature_list = list(fullData.columns)
features = np.array(fullData)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)

rs = RobustScaler()  
train_features = rs.fit_transform(train_features)  
test_features = rs.transform(test_features)  

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('merra_pm2.5')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# rf = RandomForestRegressor(n_estimators=1400, max_features='sqrt', max_depth=None, min_samples_split=5, min_samples_leaf=2,bootstrap=False)

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=16)
rf_random.fit(train_features, train_labels)
print(rf_random.best_params_)

predictions = rf_random.best_estimator_.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'ug/m3.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))

importances = list(rf_random.best_estimator_.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

