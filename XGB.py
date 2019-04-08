import numpy as np
import pandas as pd
from datetime import time
# from GetTraining import GetTraining
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, GridSearchCV
from fancyimpute import MICE
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import seaborn as sns; sns.set()
import time
import pickle
import logging
import copy


def smape(actual, forecast):
    return 100/len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

data = 'xgb'

start_time = time.time()
# fullDataList = GetTraining(data)
# pickle.dump(fullDataList, open("fullDataList-xgb.pkl.dat", "wb"))
fullDataList = pickle.load(open("fullDataList-xgb.pkl.dat", "rb"))

fullData = pd.concat(fullDataList, ignore_index=True)
fullData.eval("wind10m = sqrt(uwind**2 + vwind**2)", engine='numexpr', inplace=True)
fullData.eval("wind875 = sqrt(uwind_875**2 + vwind_875**2)", engine='numexpr', inplace=True)

print("AOD Missing %",fullData['AOD'].isnull().sum()/len(fullData)*100)  # percent missingness
print("LST Missing %",fullData['modis_lst'].isnull().sum()/len(fullData)*100)  # percent missingness

print("\nTotal valid observations: {} \n".format(len(fullData.index)))
# print(fullData.describe())

# plt.figure()
# fullData.plot(subplots=True, figsize=(15, 25))
# plt.title('Scatter Matrix')
# plt.savefig("./plots/scatterXGB.png")

# plt.figure(figsize=(17,14))
# plt.title('Correlation Heatmap')
# cor = fullData.corr(min_periods=3)  #Calculate the correlation of the above variables
# sns.heatmap(cor, square=True, cbar_kws=dict(ticks=np.arange(-1, 1, 0.2)), annot=True, cmap='coolwarm', center=0, annot_kws={"size":9})
# plt.subplots_adjust(left=0.008, right=0.993, top=0.963, bottom=0.236)
# plt.savefig("./plots/heatXGB.png")

labels = np.array(fullData['PM2.5'])
fullData = fullData.drop(['PM2.5', 'merra_pm2.5'], axis=1) 
#'vwind','uwind','uwind_875','vwind_875',

feature_list = list(fullData.columns)
features = np.array(fullData)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 2922, random_state = 42)
test_featuresA, test_featuresB, test_labelsA, test_labelsB = train_test_split(test_features, test_labels, test_size=1461, random_state=42)

mean_train = np.mean(train_labels)
baseline_predictions = np.ones(train_labels.shape) * mean_train
mae_baseline = metrics.mean_absolute_error(train_labels, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

params = {  
    # 'max_depth': [6, 9, 13, None],
    # 'min_child_weight': [1, 3, 5, 9],
    # 'gamma':[0, 0.05, 0.1 ,0.5],
    # 'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
    # 'colsample_bytree': [0.6,0.7,0.75,0.8,0.85,0.9],
    # 'reg_alpha':[0.0005, 0.001, 0.01, 0.1, 0.5],
    # 'reg_lambda':[0.01, 0.1, 1, 5, 10],
    # 'learning_rate':[0.05, 0.01, 0.005, 0.001],
} 

# xgb = XGBRegressor(random_state=42, n_jobs=32,max_depth=10,min_child_weight=2, gamma=1,subsample=0.9, colsample_bytree=0.9,learning_rate=0.01,reg_alpha=0.001,reg_lambda=5,n_estimators=5000)

# folds = 3
# param_comb = 3 

# kf = KFold(n_splits=folds, shuffle = True, random_state = 42)
# random_search = RandomizedSearchCV(xgb, params, scoring='neg_mean_absolute_error', n_iter=param_comb,random_state=42, cv=kf.split(train_features, train_labels), verbose=3,  n_jobs=32)

# random_search.fit(train_features, train_labels, early_stopping_rounds=100, eval_set=[(test_featuresB,test_labelsB)], eval_metric='mae')

# print(random_search.best_params_)
# print(random_search.best_estimator_)
# best_model = random_search.best_estimator_

# pickle.dump(best_model, open("model.pkl.dat", "wb"))

best_model = pickle.load(open("model.pkl.dat", "rb"))
print(best_model.get_params)

predictions = best_model.predict(test_featuresA)

smape_A = smape(test_labelsA, predictions)
print('sMAPE:', round(np.mean(smape_A), 2), '%.')
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labelsA, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labelsA, predictions)))
print('r2_score:', metrics.r2_score(test_labelsA, predictions))

importances = list(best_model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {:4.3f}'.format(*pair)) for pair in feature_importances]

mae = np.absolute(predictions - test_labelsA)

test_featuresA2 = copy.deepcopy(test_featuresA)
test_featuresB2 = copy.deepcopy(test_featuresB)

mice = MICE(n_imputations=75, verbose=False)
test_featuresA2 = mice.complete(test_featuresA2)
test_featuresB2 = mice.complete(test_featuresB2)

# pickle.dump(test_featuresA2, open("imp_X_test.pkl.dat", "wb"))
# test_featuresA2 = pickle.load(open("imp_X_test.pkl.dat", "rb"))

# medians = np.nanmedian(np.array(train_features), axis=0)
# inds = np.where(np.isnan(test_featuresA))
# test_featuresA2 = copy.deepcopy(test_featuresA)
# test_featuresA2[inds] = np.take(medians, inds[1])

non_nans = (~np.isnan(np.array(test_featuresA2))).sum(axis=0) # Indices [1] & [-3] have nans
print(np.where(non_nans == 0, np.nan, non_nans))


gbr_params = {
    # 'max_depth': [4,6,9,15,None],
    # 'min_samples_split': [2,3,5,8,9],
    # 'min_samples_leaf' : [1, 2, 3, 4, 8],
    # 'max_features' : [1, 2, 3, 4, 5, 6, 7],
    # 'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
    # 'min_impurity_decrease':[0, 0.25, 0.5, 0.75, 1.25],
    # 'learning_rate':[0.6,0.4,0.3,0.2,0.1, 0.05],
} #

alpha = 0.975
gbr = GradientBoostingRegressor(loss='quantile', criterion='mae', random_state=42, alpha=alpha,learning_rate=0.2, n_estimators=2000, n_iter_no_change=50, max_depth=None,min_samples_split=2,min_samples_leaf=2,max_features=5,subsample=0.8,warm_start=True, tol=0.0001)
grid = GridSearchCV(gbr, gbr_params, cv=3, n_jobs=32, scoring='neg_mean_absolute_error')
print("Fitting #1")
grid.fit(test_featuresA2, mae)
print(grid.best_params_, grid.best_score_)
best_grid = grid.best_estimator_
print(best_grid.n_estimators_)
pickle.dump(best_grid, open("y_upper_imp.pkl.dat", "wb"))

# best_grid = pickle.load(open("y_upper_imp.pkl.dat", "rb"))
# print(best_grid.get_params)
print("Predicting #1")
y_upper = best_grid.predict(test_featuresB2)

smape_up = smape(test_labelsB, y_upper)
print('sMAPE:', round(np.mean(smape_up), 2), '%.')
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labelsB, y_upper))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labelsB, y_upper)))
print('r2_score:', metrics.r2_score(test_labelsB, y_upper))

alpha2 = 0.025
gbr2 = GradientBoostingRegressor(loss='quantile', criterion='mae', random_state=42, alpha=alpha2,learning_rate=0.2, n_estimators=2000, n_iter_no_change=50, max_depth=None,min_samples_split=2,min_samples_leaf=3,max_features=1,subsample=0.8,warm_start=True, tol=0.0001)
grid2 = GridSearchCV(gbr2, gbr_params, cv=3, n_jobs=32, scoring='neg_mean_absolute_error')
print("Fitting #2")
grid2.fit(test_featuresA2, mae)
print(grid2.best_params_, grid2.best_score_)
best_grid2 = grid2.best_estimator_
print(best_grid2.n_estimators_)
pickle.dump(best_grid2, open("y_lower_imp.pkl.dat", "wb"))

# best_grid2 = pickle.load(open("y_lower_imp.pkl.dat", "rb"))
# print(best_grid2.get_params)
print("Predicting #2")
y_lower = best_grid2.predict(test_featuresB2)

smape_low = smape(test_labelsB, y_lower)
print('sMAPE:', round(np.mean(smape_low), 2), '%.')
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labelsB, y_lower))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labelsB, y_lower)))
print('r2_score:', metrics.r2_score(test_labelsB, y_lower))

test_labelsA = np.array(test_labelsA)
predictions = np.array(predictions)
y_upper = np.array(y_upper)
y_lower = np.array(y_lower)
mae = np.array(mae)
interval = np.abs(y_upper - y_lower).tolist()

sorted_labels = np.argsort(test_labelsA)
test_labelsA = test_labelsA[sorted_labels]
predictions = predictions[sorted_labels]

y_upper_pred = test_labelsA + interval
y_lower_pred = test_labelsA - interval
plt.figure(figsize=(10,10))
# plt.plot(test_labelsA, predictions, 'bx')
# plt.plot(test_labelsA, mae, 'g+')
# plt.fill_between(test_labelsA, y_lower_pred, y_upper_pred, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

fig, ax = plt.subplots(figsize=(15,15))
ax.plot(test_labelsA, predictions, 'b+')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.savefig("./plots/predictionsXGB.png")

test_labelsB = np.array(test_labelsB)

sort_ind = np.argsort(interval)
test_labelsB = test_labelsB[sort_ind]
predictions = predictions[sort_ind]
y_upper = y_upper[sort_ind]
y_lower = y_lower[sort_ind]
mae = mae[sort_ind]

mean = (y_upper + y_lower) / 2
# Center such that the mean of the prediction interval is at 0.0
test_labelsB = test_labelsB - mean
predictions = predictions - mean
y_upper = y_upper - mean
y_lower = y_lower - mean
mae = mae - mean

plt.figure(figsize=(22,11))
# plt.plot(predictions, "r+")
# plt.plot(test_labelsA, "kx")
plt.plot(mae, "gx")
plt.fill_between(
    np.arange(len(y_upper)), y_lower, y_upper, alpha=0.5, color="r",
    label="Pred. interval")
plt.xlabel("Ordered samples")
plt.ylabel("Values and prediction intervals")
plt.ylim(-25, 25)
plt.savefig("./plots/pred-int-XGB.png")

end_time = time.time()
print("\nTime: ", end_time - start_time)