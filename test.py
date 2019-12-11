#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
from joblib import load
import warnings
import logging
logging.getLogger('tsfresh').setLevel(logging.ERROR)
from tsfresh import extract_features, select_features,feature_selection
from tsfresh.utilities.dataframe_functions import impute
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[27]:


parser = argparse.ArgumentParser(description='argument parsing')
parser.add_argument('--file', type=str, default='data/test.csv')
args = parser.parse_args()
file = args.file
print("Test file successfully : {}".format(str(file)))
df = pd.read_csv(file)


# In[114]:


def test():
    print("=======================")
    print("Uploading Feature file")
    print("=======================")
    x = pd.read_csv('data/features_file.csv')
    test_file = pd.read_csv('data/test.csv')
    print("=======================")
    print("Extracting Features from test file")
    print("=======================")
    featured = feature_collection(test_file)
    X_train = pd.read_csv('train_data.csv')
    scaler = load("scaler.save")
    featured = scaler.transform(featured)
    print("=======================")
    print("Loading models")
    print("=======================")
    SVM = load('models/SVM.joblib')
    LR = load('models/Logistic.joblib')
    RFC = load('models/random_forest.joblib')
    ADA = load('models/ADA.joblib')
    print("=======================")
    print("Predicting based on four models")
    print("=======================")
    cols=['SVM','LR','RFC','ADA']
    output = pd.DataFrame(columns=cols)
    output['SVM'] = SVM.predict(featured)
    output['LR'] = LR.predict(featured)
    output['RFC'] = RFC.predict(featured)
    output['ADA'] = ADA.predict(featured)
    print("=======================")
    print("Output saved in output.csv file in current dir")
    print("=======================")
    output.to_csv('output.csv')
    print(output)
    return output


# In[115]:


def feature_collection(test_file):
    d = test_file.stack()
    d.index.rename([ 'id', 'time' ], inplace = True )
    d = d.reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = extract_features( d, column_id = "id", column_sort = "time")
    impute(f)
    assert f.isnull().sum().sum() == 0
    f=f[['0__spkt_welch_density__coeff_2', '0__fft_coefficient__coeff_1__attr_"abs"','0__partial_autocorrelation__lag_1','0__autocorrelation__lag_1','0__autocorrelation__lag_2']]
    f.to_csv('test_fe.csv')
    return f


# In[116]:

if __name__=="__main__":
    test()
