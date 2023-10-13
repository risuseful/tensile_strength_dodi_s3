# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# ----------------------------------------------
#
# import libraries
#
# ----------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import itertools
from sklearn.inspection import permutation_importance
import plotly.express as px



# read data
d = pd.read_csv('dat.csv')

# compute min and max
max_f = max(d['Feeding'])
min_f = min(d['Feeding'])
max_t = max(d['Temperature'])
min_t = min(d['Temperature'])
max_s = max(d['Speed'])
min_s = min(d['Speed'])




# set training set, input, and output
X = d[['Feeding', 'Temperature', 'Speed']].values
y = d['Strength'].values

# feature scaling
scaler_in = StandardScaler()
scaler_in.fit(X)
X = scaler_in.transform(X)

# output scaling
scaler_out = StandardScaler()
y = y.reshape(-1,1) # need to reshape in order to use scaler_out
scaler_out.fit(y)
y = scaler_out.transform(y)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# fit nnet model

nnet = MLPRegressor(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(7, 2),
                     random_state=1)


nnet.fit(X_train,y_train) # fit model




# ---------------------------------------
#
# matplotlib font size
#
# ---------------------------------------

plt.rc('axes', titlesize=14) #fontsize of the title
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
plt.rc('xtick', labelsize=14) #fontsize of the x tick labels
plt.rc('ytick', labelsize=14) #fontsize of the y tick labels

# show accuracy of prediction
fig = plt.figure(figsize=(18,6))
# ---------------------------------------



# # ---------------------------------------
# #
# # Test set prediction
# #
# #----------------------------------------

# y_pred_hat = nnet.predict(X_test) # predict

# # inverse transform
# y_pred_hat = y_pred_hat.reshape(-1,1) # reshape predicted
# y_pred = scaler_out.inverse_transform(y_pred_hat)
# y_test = y_test.reshape(-1,1) # reshape actual
# y_test = scaler_out.inverse_transform(y_test)

 

# # correlation coefficient
# r,p = pearsonr(y_pred.reshape(-1),y_test.reshape(-1))

# plt.subplot(1,3,3)
# plt.plot(y_test,y_pred,'b.', markersize=11)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# err = sqrt(mean_squared_error(y_test,y_pred))
# s = "Test set: " + r'$R^{2}$' + ' = ' + str(round(r**2,3)) + ", RMSE = " + str(round(err,3))
# plt.title(s)
# max_val = max(max(y_test), max(y_pred))
# x_red = np.arange(max_val)
# plt.plot(x_red, x_red, 'r--')




# ---------------------------------------
#
# Training + Test sets prediction
#
#----------------------------------------

y_hat = nnet.predict(X) # predict

# inverse transform
y_hat = y_hat.reshape(-1,1) # reshape predicted
y_hat = scaler_out.inverse_transform(y_hat)
y = y.reshape(-1,1) # reshape actual
y = scaler_out.inverse_transform(y)

 

# correlation coefficient
r,p = pearsonr(y.reshape(-1),y_hat.reshape(-1))

plt.subplot(1,3,3)
plt.plot(y,y_hat,'b.', markersize=11)
plt.ylabel('Actual')
plt.xlabel('Predicted')
err = sqrt(mean_squared_error(y, y_hat))
s = "Training+Test: " + r'$R^{2}$' + ' = ' + str(round(r**2,3)) + ", RMSE = " + str(round(err,3))
plt.title(s)
max_val = max(max(y), max(y_hat))
x_red = np.arange(max_val)
plt.plot(x_red, x_red, 'r--')




# ---------------------------------------------------------
# Note: Random Forests code
#
# ---------------------------------------------------------
# rgr = RandomForestRegressor(max_depth=2, random_state=0)
# rgr.fit(X,y)
# y_pred = rgr.predict(X)
# # print(rgr.predict([[0, 0, 0]]))

# result = permutation_importance(
#     rgr,
#     X,
#     y,
#     n_repeats=10,
#     random_state=42,
#     n_jobs=2)

# feature_names = list(d.columns)[0:3]
# forest_importances = pd.Series(result.importances_mean, index=feature_names)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.show()
# ---------------------------------------------------------




# -------------------------------------------------------------
#
# generate X_test from Cartesian Product of g_f, g_t, and g_s
#
# -------------------------------------------------------------
# Note:
# import itertools
# import numpy as np

# p1 = np.arange(1,4,1,dtype=int)
# p2 = np.arange(10,14,1,dtype=int)
# p3 = np.arange(57, 60, 1, dtype=int)

# P_grid = [p1, p2, p3]
# P_test = np.array([])
# for element in itertools.product(*P_grid):
#     # print(element)
#     P_test = np.append(P_test,element, axis=0)
#
#
# nr=int(len(P_test)/3)
# P_test = P_test.reshape((nr,3))
# -------------------------------------------------------------






# -------------------------------------------------------------
# generate grid of independent variables
# -------------------------------------------------------------
g_f = np.arange(min_f, max_f, 1, dtype=float)
g_t = np.arange(min_t, max_t, 5, dtype=float)
g_s = np.arange(min_s, max_s, 50,   dtype=float)
#g_s = (g_t.copy()*0.0) + np.mean(d['Speed']) # replace g_s with mean value

# top 2 important vars (Random Forests): feeding and temperature
# constant speed = mean of speed

X_grid = [g_f, g_t, g_s]
X_test = np.array([])
for element in itertools.product(*X_grid): # perform cross product
    X_test = np.append(X_test,element, axis=0)

nr = int(len(X_test)/3) # since there are 3 independent vars
X_test = X_test.reshape((nr,3))

# list unique rows in X_test; non-unique rows exist due to repeating values in g_s
X_test = np.unique(X_test, axis=0)

# scale X_test values according to defined scaler
X_test_hat = scaler_in.transform(X_test)
# -------------------------------------------------------------



# predict on X_test
# y_test = rgr.predict(X_test) # Random Forests prediction
y_test_hat = nnet.predict(X_test_hat) # Neural Net prediction
y_test_hat = y_test_hat.reshape(-1,1) # reshape to use inverse of scaler_out
y_test = scaler_out.inverse_transform(y_test_hat)
plt.plot(y_test_hat)


# # plot 3D
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(X_test[:,0], X_test[:,1], y_test)

# ax.set_xlabel('Feeding Rate')
# ax.set_ylabel('Temperature')
# ax.set_zlabel('Tensile Strength')

# plt.show()

# plot 3D with plotly
# column names: Feeding Rate (mm/min), Temperature (C), Rotational Speed (RPM), Tensile Strength (MPa)
d_3d = np.concatenate((X_test, y_test), axis=1)
dt = {'names':['Feeding Rate (mm/min)', 'Temperature (C)', 'Rotational Speed (RPM)', 'Tensile Strength (MPa)'], 'formats': [float, float, float, float]}
dp = np.zeros(len(X_test), dtype=dt)
dp['Feeding Rate (mm/min)'] = d_3d[:,0]
dp['Temperature (C)'] = d_3d[:,1]
dp['Rotational Speed (RPM)'] = d_3d[:,2]
dp['Tensile Strength (MPa)'] = d_3d[:,3]
fig = px.scatter_3d(dp, x='Feeding Rate (mm/min)', y='Temperature (C)', z='Tensile Strength (MPa)',
              color='Rotational Speed (RPM)')
fig.show()
