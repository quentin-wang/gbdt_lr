'''
lightgbm man page: 
http://lightgbm.readthedocs.io/en/latest/python/lightgbm.html#

reference(gbdt+lr example):
https://github.com/neal668/LightGBM-GBDT-LR/blob/master/GBFT%2BLR_simple.py
'''

import sys
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

import lightgbm as lgb
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

np.set_printoptions(linewidth=10000, threshold=sys.maxsize) 
# np.set_printoptions(linewidth=10000, threshold=2000) 

## build data
iris=load_iris()
iris = pd.DataFrame(load_iris().data)
iris.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
iris['Species'] = load_iris().target

## lable binary 
iris.Species=iris.Species%2


## train test split
train=iris[0:120]
test=iris[120:]
print(f"train: {train.shape}")
print(f"test: {test.shape}")

X_train=train.filter(items=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
X_test=test.filter(items=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
print(f"train: {X_train.shape}, test: {X_test.shape}")

y_train=train[[train.Species.name]]
y_test=test[[test.Species.name]]

## build lgb dataset

#########################################################
##label need reshape from (shape[0],1)  to (shape[0],)###
#########################################################
#lgb_train = lgb.Dataset(X_train.values, y_train.values.reshape(y_train.shape[0],),feature_name=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
lgb_train = lgb.Dataset(X_train.values, y_train.values.reshape(y_train.shape[0],))
lgb_eval = lgb.Dataset(X_test.values, y_test.values.reshape(y_test.shape[0],), reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 63,
    'num_trees': 100,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 63

gbm = lgb.train(params=params,train_set=lgb_train,num_boost_round=3000,valid_sets=lgb_train)

y_pred = gbm.predict(X_train,pred_leaf=True)
print(f"y_pred: {y_pred.shape}")
print(f"y_pred: {y_pred}")

## build train matrix
transformed_training_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_training_matrix[i][temp] += 1

y_pred = gbm.predict(X_test)
print(f"y_pred: {y_pred}")

y_pred = gbm.predict(X_test,pred_leaf=True)

## build test matrix
transformed_testing_matrix = np.zeros([len(y_pred),len(y_pred[0]) * num_leaf],dtype=np.int64)
for i in range(0,len(y_pred)):
	temp = np.arange(len(y_pred[0])) * num_leaf - 1 + np.array(y_pred[i])
	transformed_testing_matrix[i][temp] += 1

print('Feature importances:', list(gbm.feature_importance()))
print('Feature importances:', list(gbm.feature_importance("gain")))

# print(f"After gbdt: {transformed_training_matrix}, {transformed_training_matrix.shape}")
c = np.array([1,0.5,0.1,0.05,0.01,0.005,0.001])
for t in range(0,len(c)):
    lm = LogisticRegression(penalty='l2',C=c[t]) # logestic model construction
    lm.fit(transformed_training_matrix,y_train.values.reshape(y_train.shape[0],))  # fitting the data
    y_pred_est = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
    print(f"y_pred_est: {y_pred_est[:, 0]}")

# 随着正则的增强，置信概率在下降。错误样本的概率在上升，说明模型的泛化能力在增强。
# Feature importances: [24.448063007555902, 31.299759724421165, 474.6726898995123, 210.56304038448195]
# y_pred_est: [0.99366969 0.82852744 0.99480186 0.02653658 0.99419868 0.98660464 0.02653658 0.02327056 0.99621881 0.84229425 0.99480186 0.99453676 0.99621881 0.50048176 0.80190512 0.99366969 0.9960257  0.99023413 0.02951142 0.99366969 0.99366969 0.92325656 0.99340274 0.99366969 0.99419868 0.92325656 0.73547103 0.96597798 0.9960257  0.94050939]
# y_pred_est: [0.98910699 0.78951301 0.99070087 0.04707879 0.98987364 0.98001907 0.04707879 0.04219124 0.99273024 0.83101617 0.99070087 0.99037865 0.99273024 0.48214838 0.76180035 0.98910699 0.99247781 0.98434273 0.05057446 0.98910699 0.98910699 0.91339064 0.98801394 0.98910699 0.98987364 0.91339064 0.70905109 0.953624   0.99247781 0.92705662]
# y_pred_est: [0.96683057 0.69493765 0.96865887 0.14571157 0.96812265 0.95462624 0.14571157 0.14054209 0.97205243 0.80410414 0.96865887 0.9690298  0.97205243 0.46945235 0.66959945 0.96683057 0.97238435 0.95947567 0.15200525 0.96683057 0.96683057 0.89354151 0.96132419 0.96683057 0.96812265 0.89354151 0.65820095 0.92038864 0.97238435 0.89918536]
# y_pred_est: [0.94951319 0.66007089 0.95033368 0.21062034 0.95089494 0.9374189  0.21062034 0.20949077 0.95398642 0.78941969 0.95033368 0.95188677 0.95398642 0.47362147 0.63577192 0.94951319 0.95543064 0.94196346 0.22028095 0.94951319 0.94951319 0.88193315 0.94103044 0.94951319 0.95089494 0.88193315 0.64122153 0.90288572 0.95543064 0.88581643]
# y_pred_est: [0.87485004 0.60357652 0.87051642 0.36146583 0.87625265 0.86527636 0.36146583 0.37170708 0.87399727 0.72813607 0.87051642 0.87731153 0.87399727 0.48600991 0.57671074 0.87485004 0.88063476 0.86887685 0.37856269 0.87485004 0.87485004 0.8199107  0.85854438 0.87485004 0.87625265 0.8199107  0.60716446 0.83233626 0.88063476 0.82317687]
# y_pred_est: [0.82160953 0.58662336 0.81523517 0.41016626 0.82300172 0.81368371 0.41016626 0.42155906 0.81836069 0.69065375 0.81523517 0.82408165 0.81836069 0.49309077 0.56093791 0.82160953 0.82708927 0.81682958 0.42685039 0.82160953 0.82160953 0.77315625 0.80376911 0.82160953 0.82300172 0.77315625 0.59134935 0.78317546 0.82708927 0.77656285]
# y_pred_est: [0.67812318 0.56610991 0.67244423 0.49469192 0.67913114 0.67462928 0.49469192 0.5016093  0.67407023 0.61176559 0.67244423 0.67996945 0.67407023 0.52277584 0.55092186 0.67812318 0.68157577 0.67624965 0.5040009  0.67812318 0.67812318 0.65317774 0.66668194 0.67812318 0.67913114 0.65317774 0.56610652 0.65779992 0.68157577 0.65553839]
