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



# read data
d = pd.read_csv('dat.csv')

# compute min and max
max_f = max(d['Feeding'])
min_f = min(d['Feeding'])
max_t = max(d['Temperature'])
min_t = min(d['Temperature'])
max_s = max(d['Speed'])
min_s = min(d['Speed'])


# generate grid
g_f = np.arange(min_f, max_f, 1, dtype=float)
g_t = np.arange(min_t, max_t, 5, dtype=float)
g_s = np.arange(min_s, max_s, 25, dtype =float)

# set training set, input, and output
X = d[['Feeding', 'Temperature', 'Speed']].values
y = d['Strength'].values

rgr = RandomForestRegressor(max_depth=2, random_state=0)
rgr.fit(X,y)
y_pred = rgr.predict(X)
# print(rgr.predict([[0, 0, 0]]))

result = permutation_importance(
    rgr,
    X,
    y,
    n_repeats=10,
    random_state=42,
    n_jobs=2)

feature_names = list(d.columns)[0:3]
forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

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

X_grid = [g_f, g_t, g_s]
X_test = np.array([])
for element in itertools.product(*X_grid):
    X_test = np.append(X_test,element, axis=0)

nr = int(len(X_test)/3) # since there are 3 independent vars
X_test = X_test.reshape((nr,3))


# predict on X_test
y_test = rgr.predict(X_test)

