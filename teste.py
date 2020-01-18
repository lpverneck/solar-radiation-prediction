import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmo as pg
# import elm
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


project_dir = os.path.dirname(__file__)
data_dir = project_dir + "\\data\\raw"


df1 = pd.read_excel(data_dir+'\\BoboDioulasso.xlsx', header=1)
# df2 = pd.read_excel(data_dir+'\\Boromo.xlsx', header=1)
# df3 = pd.read_excel(data_dir+'\\BurDedougou.xlsx', header=1)
# df4 = pd.read_excel(data_dir+'\\Dori.xlsx', header=1)
# df5 = pd.read_excel(data_dir+'\\FadaNgourma.xlsx', header=1)
# df6 = pd.read_excel(data_dir+'\\Gaoua.xlsx', header=1)
# df7 = pd.read_excel(data_dir+'\\Ouahigouya.xlsx', header=1)
# df8 = pd.read_excel(data_dir+'\\Po.xlsx', header=1)


df1.columns = ['date', 'wind speed', 't max', 't min', 'humidity max',
               'humidity min', 'vpd', 'evaporation', 'solar radiation']
aux = df1['date']
df1 = df1.set_index('date')
df1['year'] = df1.index.year
df1['month'] = df1.index.month
df1['weekday name'] = df1.index.weekday_name


# data split
X_train, X_test = df1.loc['1998':'2008'], df1.loc['2009':'2012']
y_train, y_test = X_train['solar radiation'], X_test['solar radiation']
X_train = X_train.drop(['year', 'month', 'weekday name', 'solar radiation'],
                       axis=1)
X_test = X_test.drop(['year', 'month', 'weekday name', 'solar radiation'],
                     axis=1)


# SVM
reg_svm = SVR(
    kernel='rbf',
    gamma='auto',
    degree=3,
    C=1.0,
    epsilon=0.1
)
reg_svm.fit(X_train, y_train)
y_pred_svm = reg_svm.predict(X_test)
print("SVM: \n==========")
print("R²:", metrics.r2_score(y_test, y_pred_svm))
print("MSE:", metrics.mean_squared_error(y_test, y_pred_svm), "\n-----")

y_pred_svm_df = pd.DataFrame(data=y_pred_svm, index=aux[4018:])

plt.figure()
plt.plot(y_test)
plt.plot(y_pred_svm_df, color='red')
plt.title('Prediction vs Real Values')
plt.xlabel('Date')
plt.ylabel('Solar radiation')
plt.show()


# GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf']}]
#scor = ['r2', 'neg_mean_squared_error']

opt = GridSearchCV(
    estimator=reg_svm,
    param_grid=parameters,
    scoring='r2',
    n_jobs=-1,
    iid=True,
    #refit=False
)
opt = opt.fit(X_train, y_train)

best_scores = opt.cv_results_()
best_parameters = opt.best_params_


# ANN
reg_ann = MLPRegressor(
    hidden_layer_sizes=(7, 14, 7),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='constant'
)
reg_ann.fit(X_train, y_train)
y_pred_ann = reg_ann.predict(X_test)
print("ANN: \n==========")
print("R²:", metrics.r2_score(y_test, y_pred_ann))
print("MSE:", metrics.mean_squared_error(y_test, y_pred_ann), "\n-----")


# GB
reg_gb = GradientBoostingRegressor(
    loss='ls',
    learning_rate=0.1,
    n_estimators=1000
)
reg_gb.fit(X_train, y_train)
y_pred_gb = reg_gb.predict(X_test)
print("GB: \n==========")
print("R²:", metrics.r2_score(y_test, y_pred_gb))
print("MSE:", metrics.mean_squared_error(y_test, y_pred_gb), "\n-----")


# ELM - stand by


# XGB
reg_xgb = XGBRegressor(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=1000,
    gamma=0,
    objective="reg:squarederror"
)
reg_xgb.fit(X_train, y_train)
y_pred_xgb = reg_xgb.predict(X_test)
print("XGB: \n==========")
print("R²:", metrics.r2_score(y_test, y_pred_xgb))
print("MSE:", metrics.mean_squared_error(y_test, y_pred_xgb), "\n-----")


# DE
prob = pg.problem(pg.rosenbrock(dim=3))
print(prob)
algo = pg.algorithm(pg.de(gen=500))
algo.set_verbosity(500)
pop = pg.population(prob, 20)
pop = algo.evolve(pop)

uda = algo.extract(pg.de)
uda.get_log()

# plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color = 'blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()

# rawdata.loc['1999-09-28']
# rawdata['Solar Radiation'].plot(linewidth=0.3)
# plt.xlabel('Time')
# plt.ylabel('Solar Radiation')
