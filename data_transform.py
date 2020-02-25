# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import data_prep

path = '../res/creditcarddataset.csv'
def load_data(path):
    dataframe = pd.read_csv(path)
    dataframe= dataframe.sample(frac=1,random_state=42).reset_index(drop=True)
    return  dataframe

df = load_data(path)
df.describe()
def split_dataset(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)
    return x_train, x_test, y_train, y_test

X_train, X_test, y_train, y_test = split_dataset(df)
# =============================================================================
# Feature Selection for Numeric Features
# =============================================================================
training_data= pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)],axis=1)
data_prep.plot_corr(training_data)
del X_train[' current_balance']
del X_test[' current_balance']
#del X_train[' residence_since']
#del X_test[' residence_since']

def convert_cat2float(X_train,cat_range):
    le=LabelEncoder()
    for i in cat_range:
        X_train.iloc[:,i] = le.fit_transform(X_train.iloc[:,i])
        pickle.dump(le,open('../res/le/'+str(i)+'.pkl','wb'))
    return X_train
            
def apply_le(X_test,cat_range):
    for i in cat_range:
        le = pickle.load(open('../res/le/'+str(i)+'.pkl','rb'))
        X_test.iloc[:,i]= le.transform(X_test.iloc[:,i])
    return X_test

def transform_classes(y_train):
    le=LabelEncoder()
    y_train = le.fit_transform(y_train)
    pickle.dump(le,open('../res/le/label.pkl','wb'))
    return y_train 

def apply_classes(y_test):
    le = pickle.load(open('../res/le/label.pkl','rb'))
    return le.transform(y_test)

# =============================================================================
# provide categorical feature index by removing num indexes from total
# =============================================================================
#numerical_attr=[1,6,10,13,15]
numerical_attr=[1,6,9,11,14,16]
cat_range =np.delete(list(range(0,X_train.shape[1])),numerical_attr)
X_train=convert_cat2float(X_train,cat_range)
X_test = apply_le(X_test,cat_range)

# =============================================================================
# Save all your work as numpy array.
# =============================================================================
np.save(open('../res/X_train.npy','wb'),X_train)
np.save(open('../res/X_test.npy','wb'),X_test)
np.save(open('../res/y_train.npy','wb'),transform_classes(y_train))
np.save(open('../res/y_test.npy','wb'),apply_classes(y_test))