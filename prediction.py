#!/usr/bin/env python
# coding: utf-8

# In[11]:



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis,moment
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score, classification_report
from sklearn.linear_model import LinearRegression


# In[8]:


def preprocessing(train_data, test_data):
    
    train_data['origination_date'] = pd.to_datetime(train_data['origination_date'])

    train_data['first_payment_date'] = pd.to_datetime(train_data['first_payment_date'])

    date_test= test_data['first_payment_date'].str.split('-',n=1,expand=True)
    date_test[0]=date_test[0].replace('Apr', '04')
    date_test[0]=date_test[0].replace('Mar', '03')
    date_test[0]=date_test[0].replace('Feb', '02')
    date_test[0]=date_test[0].replace('May', '05')



    date_test_formated = date_test[0]+'/'+'20'+date_test[1]

    test_data['first_payment_date']= date_test_formated
    test_data['first_payment_date'] = pd.to_datetime(test_data['first_payment_date'])
    test_data['origination_date'] = pd.to_datetime(test_data['origination_date'])

    column_1 = train_data['origination_date']

    df1 = pd.DataFrame({
                  "org_month": column_1.dt.month,
                  "org_week": column_1.dt.week,
                  "org_dayofweek": column_1.dt.dayofweek,
                  "org_weekday": column_1.dt.weekday,
                 })
    df1.head()

    column_2 = test_data['origination_date']

    df2 = pd.DataFrame({
                  "org_month": column_2.dt.month,
                  "org_week": column_2.dt.week,
                  "org_dayofweek": column_2.dt.dayofweek,
                  "org_weekday": column_2.dt.weekday,
                 })
    df2.head()

    train = pd.concat([train_data,df1],axis=1)
    train = train.drop('origination_date',axis=1)
    test = pd.concat([test_data,df2],axis=1)
    test = test.drop('origination_date',axis=1)

    column_1 = train_data['first_payment_date']

    df1 = pd.DataFrame({
                  "frst_month": column_1.dt.month,
                  "frst_week": column_1.dt.week,
                  "frst_dayofweek": column_1.dt.dayofweek,
                  "frst_weekday": column_1.dt.weekday,
                 })
    df1.head()

    column_2 = test_data['first_payment_date']

    df2 = pd.DataFrame({
                  "frst_month": column_2.dt.month,   
                  "frst_week": column_2.dt.week,
                  "frst_dayofweek": column_2.dt.dayofweek,
                  "frst_weekday": column_2.dt.weekday,
                 })
    df2.head()

    train = pd.concat([train,df1],axis=1)
    train = train.drop('first_payment_date',axis=1)
    test = pd.concat([test,df2],axis=1)
    test = test.drop('first_payment_date',axis=1)


    train['source'] = train['source'].astype("category")
    train['source'] = train['source'].cat.codes
    test['source'] = test['source'].astype("category")
    test['source'] = test['source'].cat.codes

    train["financial_institution"] = train["financial_institution"].astype('category')
    train["financial_institution"] = train["financial_institution"].cat.codes

    test["financial_institution"] = test["financial_institution"].astype('category')
    test["financial_institution"] = test["financial_institution"].cat.codes

    train['loan_purpose'] = train['loan_purpose'].astype("category")
    train['loan_purpose'] = train['loan_purpose'].cat.codes
    
    test['loan_purpose'] = test['loan_purpose'].astype("category")
    test['loan_purpose'] = test['loan_purpose'].cat.codes
    
 
    
    return train,test


# In[3]:


def feature_engineering(train,test):
    def add_trend_feature(arr, abs_values=False):
        idx = np.array(range(len(arr)))
        if abs_values:
            arr = np.abs(arr)
        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)
        return lr.coef_[0]
    trend_train=[]
    for i in range(len(train)):
        trend_train.append(add_trend_feature(train.iloc[i,0:12]))

    trend_test=[]
    for i in range(len(test)):
        trend_test.append(add_trend_feature(test.iloc[i,0:12]))
    deli =test.iloc[:,0:0+12].values

    test['max_del'] = np.max(deli,axis=1)
    test['min_del'] = np.min(deli,axis=1)
    test['mean_del'] = np.mean(deli,axis=1)
    test['non_del_months'] = np.count_nonzero(deli,axis=1)
    test['var_del'] = np.var(deli,axis=1)
    test['skewness'] = skew(deli,axis=1)
    test['kurt'] = kurtosis(deli,axis=1)
    test['sum'] = np.sum(deli,axis=1)
    test['max_min_diff'] = np.max(deli,axis=1) - np.min(deli,axis=1)
    test['median'] = np.median(deli,axis=1)
    test['trend']= trend_test



    deli = train.iloc[:,0:0+12].values
    train['max_del'] = np.max(deli,axis=1)
    train['min_del'] = np.min(deli,axis=1)
    train['mean_del'] = np.mean(deli,axis=1)
    train['non_del_months'] = np.count_nonzero(deli,axis=1)
    train['var_del'] = np.var(deli,axis=1)
    train['skewness'] = skew(deli,axis=1)
    train['kurt'] = kurtosis(deli,axis=1)
    train['sum'] = np.sum(deli,axis=1)
    train['max_min_diff'] = np.max(deli,axis=1) - np.min(deli,axis=1)
    train['median'] = np.median(deli,axis=1)
    train['trend']= trend_train
    deli = train.iloc[:,0:12]
    train = train.drop(deli,axis=1)
    train = pd.concat([train,deli],axis=1)

    deli = test.iloc[:,0:12]
    test = test.drop(deli,axis=1)
    test = pd.concat([test,deli],axis=1)
    return train,test


# In[9]:


def data_for_model(train,test):
    x_train=train.drop(['m13'],axis=1)
    y_train = train['m13']

    x_test = test

    X=np.array(x_train)
    Y=np.array(y_train)
    X_test = np.array(x_test)
    ##Train Valid split
    x_train , x_valid, y_train, y_valid = train_test_split(X,Y,test_size=0.4,random_state=1,shuffle=True,stratify=Y)
    
    x_train = np.reshape(x_train,(x_train.shape[0],1,23))
    x_valid = np.reshape(x_valid,(x_valid.shape[0],1,23))
    X_test = np.reshape(X_test,(X_test.shape[0],1,23))
    y_train_cat = to_categorical(y_train,num_classes=2)
    return x_train, y_train_cat,x_valid, y_valid, X_test


# In[7]:


def train_model(x_train, y_train,x_valid, y_valid):
    model = Sequential()
    model.add(GRU(32, input_shape=(None, 23)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2))

    model.summary()

    # Compile and fit model
    model.compile(optimizer=Adam(lr=0.0005), loss="mae")
    model.fit(x_train,y_train,epochs=20,batch_size=32,verbose=1,class_weight={0 : 1,1: 7})
    
    predictions = model.predict(x_valid)
    pred = np.argmax(predictions,axis=1)
    cnf_mat = confusion_matrix(y_valid,pred)
    print('Confusion_matrix:\n',cnf_mat)
    print('Accuracy:',accuracy_score(y_valid,pred))
    print('F1-score:',f1_score(y_valid,pred))
    print('Classification Report:\n',classification_report(y_valid,pred))
    
    return model


# In[6]:


def test_and_submit(model):
    predi = model.predict(X_test)
    predi = np.argmax(predi,axis=1)
    sub = pd.read_csv('sample_submission.csv')
    sub.head()

    sub['m13'] = predi
    sub.head()

    sub.to_csv('submission_gru_med_0.525(1,7)e-20.csv',index=False)


# In[10]:


if __name__ == "__main__":

    
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    print('--------Loaded_data---------\n')
    print('--------Data Preprocessing started---------\n')
    
    
    train , test = preprocessing(train_data,test_data)
    train =train.iloc[:,14:14+13]
    test = test.iloc[:,14:14+12]
    print('--------Feature Engineering started---------\n')
    train,test = feature_engineering(train,test)
    print('--------Training_the_model---------\n')
    x_train, y_train,x_valid, y_valid, X_test = data_for_model(train,test)
    del train_data, test_data, train, test
    model = train_model(x_train, y_train,x_valid, y_valid)
    print('--------Test and Submission---------\n')
    test_and_submit()
    print('-------THE END-----------')





