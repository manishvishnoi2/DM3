#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from joblib import load
import warnings
import logging
logging.getLogger('tsfresh').setLevel(logging.ERROR)
from tsfresh import extract_features, select_features,feature_selection
from tsfresh.utilities.dataframe_functions import impute
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[2]:


parser = argparse.ArgumentParser(description='argument parsing')
parser.add_argument('--file', type=str, default='data/test.csv')
args = parser.parse_args()
file = args.file
print("Test file successfully uploaded: {}".format(str(file)))
test_file = pd.read_csv(file)


# In[ ]:


def test():
    print("=======================")
    print("Uploading Feature file")
    print("=======================")
    print("=======================")
    print("Extracting Features from test file")
    print("=======================")
    featured = feature_collection(test_file)
    print(featured.shape)
    scaler = load("scaler.save")
    featured = scaler.transform(featured)
    print("=======================")
    print("Loading models")
    print("=======================")
    kmeans_pro = load("kmeans.bin")
    print("=======================")
    print("Predicting based on four models")
    print("=======================")
    y_test_df = pd.DataFrame(featured)
    y_test_df['Predicted'] = kmeans_pro.predict(featured)
    carb_matrix = ["10-20","20-30","30-40","40-50","60-70","50-60","70-80","80-90","90-100","0-10"]
    final = []
    for x in y_test_df.iterrows():
        final.append(carb_matrix[np.int64(x[1]['Predicted'])])
    y_test_df['carb_values'] = final
    print("=======================")
    print("Output saved in output.csv file in current dir")
    print("=======================")
    y_test_df.to_csv('output.csv')
    return y_test_df


# In[ ]:


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
    return f


# In[116]:

if __name__=="__main__":
    test()

