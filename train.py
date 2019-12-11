#!/usr/bin/env python
# coding: utf-8

# In[21]:


import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tsfresh').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    import pandas as pd
    import numpy as np
    from pprint import pprint
    from sklearn.impute import SimpleImputer
    from tsfresh import extract_features, select_features,feature_selection
    from tsfresh.utilities.dataframe_functions import impute
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from joblib import load, dump
    from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy, precision_score, recall_score, f1_score
    from sklearn.svm import SVC as SVM
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.ensemble import AdaBoostClassifier as ABC
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier as DTC
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm
    from sklearn import preprocessing
    from sklearn.linear_model import LogisticRegressionCV
    from scipy.stats import randint as sp_randint
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier


# In[22]:



def train():
    result = pd.DataFrame()
#     Extracting Data for meal class

    for x in range(5):
        d = pd.read_csv('data/mealData'+str(x+1)+'.csv', header = None,error_bad_lines=False)
        # d['y']= 1
        result = pd.concat([result,d])
    
#     Extracting data for no meal class
    # for x in range(5):
    #     d = pd.read_csv('data/Nomeal'+str(x+1)+'.csv', header = None,error_bad_lines=False)
    #     d['y']= 0
    #     result = pd.concat([result,d])
        
#         Imputing for NaN value removal
    result = impute_data(result)
    
#     Renaming Target column to dataframe
    result = pd.DataFrame(result)
    columns = list(result.columns)
    columns.pop()
    columns.append('target')
    result.columns = columns
    
#     Extracting features and writing into files
    features = feature_extract(result,'data/features_file.csv')
#     Cross fold validation and Training models
    # classifier()


# In[23]:


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
    


# In[24]:


def impute_data(result):
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    imp_mean.fit(result)
    return(imp_mean.transform(result))


# In[25]:


def classifier():
    data = pd.read_csv("data/features_file.csv")
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data['y'], test_size=0.33, random_state=42)
    X_train.to_csv('train_data.csv')
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler,"scaler.save")
    SVM(X_train,X_test,y_train,y_test)
    LR(X_train,X_test,y_train,y_test)
    RFC(X_train,X_test,y_train,y_test)
    ADA(X_train,X_test,y_train,y_test)


# In[26]:


def SVM(X_train,X_test,y_train,y_test):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=10)
    clf.fit(X_train, y_train)
    best = clf.best_estimator_
    best = load('models/SVM.joblib')
    y_pred = best.predict(X_test)
    # test_file = pd.read_csv('test_fe.csv')
#     print(best.predict(test_file))
    print("AUC of SVM {:.4f}, Accuracy of SVM is {:.4f}, Precision is {:.4f}, Recall is {:.4f}, F1 score is {:.4f}".format(AUC(y_test,y_pred), accuracy(y_test,y_pred),precision_score(y_test, y_pred), recall_score(y_test, y_pred),f1_score(y_test,y_pred) ))
    return best


# In[27]:


def LR(X_train,X_test,y_train,y_test):
    lr = LogisticRegressionCV(cv=5, random_state=0, multi_class='auto').fit(X_train, y_train)
    lr=load('models/Logistic.joblib')
    y_pred = lr.predict(X_test)
    print("AUC LR {:.4f}, Accuracy of LR is {:.4f}, Precision is {:.4f}, Recall is {:.4f}, F1 score is {:.4f}".format(AUC(y_test,y_pred), accuracy(y_test,y_pred),precision_score(y_test, y_pred), recall_score(y_test, y_pred),f1_score(y_test,y_pred) ))
    


# In[28]:


def RFC(X_train,X_test,y_train,y_test):
    clf = RandomForestClassifier(n_estimators=20)
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 2, 4],
                  "min_samples_split": [2, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
    grid_search.fit(X_train, y_train)
    grid_search=load('models/random_forest.joblib')
    y_pred = grid_search.predict(X_test)
    print("AUC of RFC {:.4f}, Accuracy of RFC is {:.4f}, Precision is {:.4f}, Recall is {:.4f}, F1 score is {:.4f}".format(AUC(y_test,y_pred), accuracy(y_test,y_pred),precision_score(y_test, y_pred), recall_score(y_test, y_pred),f1_score(y_test,y_pred) ))


# In[29]:


def ADA(X_train,X_test,y_train,y_test):
    param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "n_estimators": [1, 2]
                 }

    # DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)
    dtc = DecisionTreeClassifier(max_depth = 12)
    ABC = AdaBoostClassifier(base_estimator = dtc)

    # run grid search
    grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, cv=5)
    grid_search_ABC.fit(X_train, y_train)
    grid_search_ABC = load('models/ADA.joblib')
    y_pred = grid_search_ABC.predict(X_test)
    print("AUC of ADA {:.4f}, Accuracy of ADA is {:.4f}, Precision is {:.4f}, Recall is {:.4f}, F1 score is {:.4f}".format(AUC(y_test,y_pred), accuracy(y_test,y_pred),precision_score(y_test, y_pred), recall_score(y_test, y_pred),f1_score(y_test,y_pred) ))
    return grid_search_ABC


# In[30]:
def main():
  train()  

# train()
if __name__=="__main__":
    main()

# In[ ]:




