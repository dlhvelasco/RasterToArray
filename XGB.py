import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import time
from GetTraining import GetTraining

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn import metrics
from xgboost.sklearn import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import time
import pickle

start_time = time.time()
fullDataList = GetTraining()
fullData = pd.concat(fullDataList, ignore_index=True)
print("\nTotal valid observations: {} \n".format(len(fullData.index)))
# print(fullData.describe())


labels = np.array(fullData['PM2.5'])
fullData = fullData.drop(['PM2.5'], axis=1)

feature_list = list(fullData.columns)
features = np.array(fullData)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)

rs = RobustScaler(quantile_range = (0.1,0.9))  
train_features = rs.fit_transform(train_features)  
test_features = rs.transform(test_features)  

df_train_features = pd.DataFrame(train_features, columns=feature_list)
plt.figure()
fig, ax1 = plt.subplots(1,1, figsize=(12, 14))
ax1.set_title('After Standard Scaler')
for feat in feature_list:
    sns.kdeplot(df_train_features[feat], ax=ax1)
ax.set_xlim(-1.1,1.1)
plt.savefig("/home/dwight.velasco/scratch1/THESIS/RasterToArray/plots/dist.png")


dtrain = xgb.DMatrix(train_features, label=train_labels)
dtest = xgb.DMatrix(test_features, label=test_labels)

mean_train = np.mean(train_labels)
baseline_predictions = np.ones(train_labels.shape) * mean_train
mae_baseline = metrics.mean_absolute_error(train_labels, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

params = {  'max_depth': [3,5,10,15],
            'min_child_weight': [1,3,5,8],
            'eta':[0.2, 0.1, 0.01, 0.005],
            'subsample': [0.6,0.8,1.0],
            'colsample_bytree': [0.6,0.8,1.0],
            'n_estimators':[200,600,1000,1400,2000],
            'gamma':[0.0,0.2,0.4]
}

xgb = XGBRegressor()

folds = 5
param_comb = 5

kf = KFold(n_splits=folds, shuffle = True, random_state = 1001)
random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='neg_mean_absolute_error', cv=kf.split(train_features, train_labels), verbose=3, random_state=42, n_jobs=16)

random_search.fit(train_features, train_labels)

print(random_search.best_params_)
print(random_search.best_estimator_)

# pickle.dump(random_search.best_estimator_, open("model6675.pkl.dat", "wb"))
# loaded_model = pickle.load(open("model6675.pkl.dat", "rb"))

predictions = random_search.best_estimator_.predict(test_features)
# predictions = loaded_model.predict(test_features)
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
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))

end_time = time.time()
print("\nTime: ", end_time - start_time)

importances = list(random_search.best_estimator_.feature_importances_)
# importances = list(loaded_model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]