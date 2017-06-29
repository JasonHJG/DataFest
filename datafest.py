# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt


mydata = pd.read_table("/Users/jingang/Desktop/ASADataFest2017 Data/data.txt",nrows=100000)
dest = pd.read_table("/Users/jingang/Desktop/ASADataFest2017 Data/dest.txt",nrows=100000)

# break into smaller data 1/100 * 10884539
np.random.seed(1024)
row,col = mydata.shape
sample_size = int(row*0.01)
#create index
idx = range(row)
#choose 1/100 of population randomly
sample_idx = np.random.choice(row,sample_size, replace=False)
sample_df = mydata.loc[sample_idx]
sample_df = sample_df.set_index([range(len(sample_df))])
print(sample_df)

# create new features

# browsing time before travel
date_view = pd.to_datetime(sample_df['date_time'])
date_checkin = pd.to_datetime(sample_df['srch_ci'])
date_checkout = pd.to_datetime(sample_df['srch_co'])

sample_df['days_before_stay'] = (date_checkin - date_view).astype('timedelta64[h]')
sample_df['length_of_stay'] = (date_checkout - date_checkin).astype('timedelta64[h]')


# add destination user reviews
sample_df = sample_df.join(dest.set_index('srch_destination_id'), on='srch_destination_id')

# split into test and train
train_size = int(sample_size*0.7)  #train_size=700
idx = range(sample_size)      #[0,1,2...699]
np.random.seed(1024)
train_idx = np.random.choice(sample_size,train_size, replace=False) #choose 700 from 1000
############################################################
test_idx = np.delete(idx,train_idx)

train_df = sample_df.loc[train_idx]
test_df = sample_df.loc[test_idx]
train_df = train_df.set_index([range(len(train_df))])
test_df = test_df.set_index([range(len(test_df))])      


print(train_df.shape)
print(test_df.shape)

# select features
features_to_use = [#"date_time", 
                   "user_location_latitude", 
                   "user_location_longitude", "orig_destination_distance",
                   #"srch_ci", "srch_co",
                   "srch_adults_cnt", "srch_children_cnt",
                   "srch_rm_cnt","cnt",
                   'days_before_stay','length_of_stay']

dest_features = list(dest)
features_to_use += dest_features


categorical = ["site_name","user_location_country","user_location_region",
               "user_location_city","user_id","is_mobile","is_package",
               "channel","srch_destination_id","hotel_country","hotel_id",
               "prop_is_branded","prop_starrating","distance_band",
               "hist_price_band","popularity_band","srch_destination_name"]

for f in categorical:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[f].values) + list(test_df[f].values))
    train_df[f] = lbl.transform(list(train_df[f].values))
    test_df[f] = lbl.transform(list(test_df[f].values))
    features_to_use.append(f)
    

train_X = sparse.csr_matrix(train_df[features_to_use])
test_X = sparse.csr_matrix(test_df[features_to_use])




train_y = train_df['is_booking']

cv_scores = []
param_log = []

def runXGB(train_X, train_y, test_X, eval_result, test_y=None, feature_names=None, seed_val=0, num_rounds=2000):
    param = {}
    param['objective'] = 'binary:logistic'
    param['booster'] = 'gbtree'
    param['eta'] = 0.02
    param['max_depth'] = 4
    param['eval_metric'] = "logloss"
    param['min_child_weight'] = 2 #space['min_child_weight']  #3
    param['max_delta_step'] = 1
    param['subsample'] = 0.8 #space['subsample']
    param['colsample_bytree'] = 0.5 #space['colsample_bytree']
    param['seed'] = seed_val
    #param['lambda'] = 1# space['lambda']
    param['gamma'] = 1#space['gamma']  #0
    param['alpha'] = 1# space['alpha']
    #param['scale_pos_weight'] = 1
    num_rounds = num_rounds

    plst = list(param.items())
    param_log.append(plst)
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, evals_result= eval_result, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    pred_test_y = model.predict(xgtest)
    
    return pred_test_y, model



kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=201609)
result = 0
eval_result = {}
for dev_index, val_index in kf.split(range(train_X.shape[0])):
    dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    preds, model = runXGB( dev_X, dev_y, val_X, eval_result, val_y)
    result = log_loss(val_y, preds)
    cv_scores.append(result)
    print(cv_scores)
    break


epochs = len(eval_result['test']['logloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, eval_result['test']['logloss'], label='Train')
ax.plot(x_axis, eval_result['train']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss', fontsize = 15)
plt.title('XGBoost Log Loss',fontsize = 18)
plt.show()
plt.gcf().savefig('log_loss.png')



importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

for i in range(len(df.feature)):
    sub_str = df.feature[i][1:]
    idx = int(sub_str)
    df.feature[i] = features_to_use[idx]
    

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(30, 20),fontsize=20)
plt.title('XGBoost Feature Importance',fontsize=30)
plt.xlabel('relative importance',fontsize=25)
plt.ylabel('',fontsize=25)
plt.gcf().savefig('feature_importance_xgb.png')



preds, model = runXGB(train_X, train_y,test_X,eval_result,num_rounds=2000)
out_df = pd.DataFrame(preds)
test_y = test_df['is_booking']

log_loss(test_y, out_df) 