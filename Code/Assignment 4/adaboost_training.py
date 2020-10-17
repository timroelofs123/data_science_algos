import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

filename_train = 'C:\\Users\\leovu\\Desktop\\framingham.csv'  # csv file path with the preprocessed features and ACTIVE column
target = 'ACTIVE'
IDcol = 'INDEX'
df_train = pd.read_csv(filename_train)
predictors = [x for x in df_train.columns if x not in [target, IDcol]]
X_train = np.array(df_train.loc[:, predictors])
scaler = MinMaxScaler()
X_train_std = scaler.fit_transform(X_train)
y_train = np.array(df_train.loc[:, target])

filename_validation = 'C:\\Users\\leovu\\Desktop\\framingham.csv'  # csv file path with the preprocessed features and ACTIVE column
df_validation = pd.read_csv(filename_validation)
X_validation = np.array(df_validation.loc[:, predictors])
X_validation_std = scaler.transform(X_validation)
y_validation = np.array(df_validation.loc[:, target])


dtr = DecisionTreeRegressor(max_depth=7)
abr = AdaBoostRegressor(n_estimators=50, base_estimator=dtr, learning_rate=1)

abr.fit(X_train_std, y_train)
y_pred = abr.predict(X_validation)

print(roc_auc_score(y_validation, y_pred))
