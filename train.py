#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.impute import SimpleImputer
from tsfresh import extract_features, select_features,feature_selection
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from joblib import load, dump
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tsfresh').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
from sklearn.cluster import KMeans


# In[2]:


def train():
    result = pd.DataFrame()
    for x in range(5):
            d = pd.read_csv('data/mealData'+str(x+1)+'.csv', header = None,error_bad_lines=False)
            f = pd.read_csv('data/mealAmountData'+str(x+1)+'.csv', header = None,error_bad_lines=False)
            f.values.tolist()
            final_val =[]
            for x in range(d.shape[0]):
                final_val.append(f[0][x])
            d['carbo'] = final_val
            result = pd.concat([result,d])
    result = impute_data(result)
    result = pd.DataFrame(result)
    columns = list(result.columns)
    columns.pop()
    columns.append('target')
    result.columns = columns
    features = feature_extract(result,'data/features_file.csv')
    data = pd.read_csv("features_file.csv")
    data = data.iloc[:, :-1]
    data = data.drop(['id'],axis=1)
    print(data.head())
    scaler = MinMaxScaler().fit(data)
    data = scaler.transform(data)
    print(data.shape)
    dump(scaler,"scaler.save")
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
    dump(kmeans,"kmeans.bin")


# In[3]:


def impute_data(result):
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    imp_mean.fit(result)
    return(imp_mean.transform(result))


# In[4]:


def feature_extract(result, filename):
    y = result.target
    result.drop( 'target', axis = 1, inplace = True )
    d = result.stack()
    d.index.rename([ 'id', 'time' ], inplace = True )
    d = d.reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = extract_features( d, column_id = "id", column_sort = "time")
    impute(f)
    assert f.isnull().sum().sum() == 0
    f=f[['0__spkt_welch_density__coeff_2', '0__fft_coefficient__coeff_1__attr_"abs"','0__partial_autocorrelation__lag_1','0__autocorrelation__lag_1','0__autocorrelation__lag_2']]
    f['y'] = y  
    f.to_csv( filename, index = None )
    return f


# In[5]:


if __name__=="__main__":
    train()
