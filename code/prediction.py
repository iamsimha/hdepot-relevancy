from __future__ import division
from sklearn import preprocessing

import pandas as pd
import numpy as np
import sklearn as sk
import xgboost as xgb
import pickle
import inspect

import utils
np.random.seed(123)

def round_to_nearest(num):
    rel = np.array([3.00,2.33,2.67,2.00,1.67,1.33,1.00])
    return rel[np.argmin(np.abs(rel - num))]


def xgb_boost_model():
    df_all = pickle.load(open("../output/features/basic_features.pkl", 'r'))
    test_ind = df_all.relevance == -1
    test_data = df_all[test_ind]
    train_data = df_all[~test_ind]
    test_data = test_data.drop(['relevance'], axis=1)
    le = preprocessing.LabelEncoder()
    le.fit(train_data['relevance'])

    ids = test_data['id']

    train, test, hold_out = utils.split_dataset(train_data)

    relevant_columns =['title_similarity', 'product_desc_similarity',  'title_similarity_common', 'product_desc_similarity_common', 'description_length', 'search_length']
    dTrain = xgb.DMatrix(train['X'][relevant_columns], label=train['Y'])
    dTest = xgb.DMatrix(test['X'][relevant_columns], label=test['Y'])
    dHold_out = xgb.DMatrix(hold_out['X'][relevant_columns], label=hold_out['Y'])
    dSubmit = xgb.DMatrix(test_data[relevant_columns])

    param = {'bst:max_depth':5  , 'bst:eta':0.05, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse'}

    evallist = [(dTest, 'eval'), (dTrain, 'train')]
    numRound = 200
    bst = xgb.train(param, dTrain, numRound, evallist)

    predHoldout = bst.predict(dHold_out)
    print "Mean square hold out error ", utils.rmse(hold_out['Y'], predHoldout)

    predY = bst.predict(dSubmit)
    utils.debug_model(hold_out['X'], hold_out['Y'], predY)
    #pd.DataFrame({"id": ids, "relevance": predY}).to_csv("../output/submissions/"+inspect.currentframe().f_code.co_name + '_s.csv',index=False)

def main():
    xgb_boost_model()
if __name__ == '__main__':
    main()
