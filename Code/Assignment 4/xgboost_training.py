from xgboost import XGBRegressor, cv, DMatrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

filename = 'C:\\Users\\leovu\\Desktop\\framingham.csv'  # csv file path with the preprocessed features and ACTIVE column
target = 'ACTIVE'
IDcol = 'INDEX'
df = pd.read_csv(filename)
predictors = [x for x in df.columns if x not in [target, IDcol]]
X = np.array(df.loc[:, predictors])
scaler = MinMaxScaler()
X_std = scaler.fit_transform(X)
y = np.array(df.loc[:, target])

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.1, random_state=0, stratify=y)

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())


parameters_range = {'objective': ['binary:logistic'],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators': [20, 30, 40, 50, 60],

                    'max_depth': [5, 6, 7, 8, 9, 10],
                    'min_child_weight': [1],
                    'gamma': [0, 0.1, 0.2],
                    'subsample': [1],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],

                    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
                    }

parameters = {'objective': ['binary:logistic'],
                'learning_rate': [0.2],
                'n_estimators': [1000],

                'max_depth': [5, 6, 7, 8, 9, 10],
                'min_child_weight': [1],
                'gamma': [0, 0.1, 0.2],
                'subsample': [1],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],

                'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
                }


def grid_search(parameters, X_train_res, y_train_res, X_test, y_test, useTrainCV=False):
    xgbmodel = XGBRegressor()
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    grid_search_xg = GridSearchCV(xgbmodel, parameters, scoring='roc_auc', n_jobs=1, cv=kfold, verbose=1)
    result_gcv_xgb = grid_search_xg.fit(X_train_res, y_train_res)
    best_params = result_gcv_xgb.best_params_
    print("Best params: %s" % (best_params))

    # rebuild using best params
    xg_reg = XGBRegressor(objective=best_params['objective'], learning_rate=best_params['learning_rate'],
                          max_depth=best_params['max_depth'], n_estimators=best_params['n_estimators'],
                          min_child_weight=best_params['min_child_weight'], gamma=best_params['gamma'],
                          colsample_bytree=best_params['colsample_bytree'], subsample=best_params['subsample'],
                          reg_alpha=best_params['reg_alpha'])

    if useTrainCV:
        xgb_param = xg_reg.get_xgb_params()
        xgtrain = DMatrix(X_train_res, label=y_train_res)
        cvresult = cv(xgb_param, xgtrain, num_boost_round=xg_reg.get_params()['n_estimators'], folds=kfold,
                          metrics='auc', early_stopping_rounds=20)
        xg_reg.set_params(n_estimators=cvresult.shape[0])
        print("Best number of estimators: %i" % (cvresult.shape[0]))

    eval_set = [(X_test, y_test)]
    xg_reg.fit(X_train_res, y_train_res, eval_metric="error", eval_set=eval_set, verbose=False)
    y_pred_train = xg_reg.predict(X_train_res)
    #print("Accuracy train: %f" % (accuracy_score(y_train_res, y_pred_train)))
    #print("Recall train: %f" % (recall_score(y_train_res, y_pred_train)))
    #print("Precision train: %f" % (precision_score(y_train_res, y_pred_train)))
    print("AUC train: %f" % (roc_auc_score(y_train_res, y_pred_train)))
    y_pred = xg_reg.predict(X_test)
    #print("Accuracy test: %f" % (accuracy_score(y_test, y_pred)))
    #print("Recall test: %f" % (recall_score(y_test, y_pred)))
    #print("Precision test: %f" % (precision_score(y_test, y_pred)))
    print("AUC test: %f" % (roc_auc_score(y_test, y_pred)))


# testing
grid_search(parameters, X_train_res, y_train_res, X_test, y_test, useTrainCV=True)
