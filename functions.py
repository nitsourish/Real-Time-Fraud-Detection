import psycopg2 as pg
import yaml
from pathlib import Path
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tqdm
import cv2
from scipy.spatial import distance
from PIL import Image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,cross_validate,GridSearchCV,RandomizedSearchCV
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import scipy
import datetime
import seaborn as sns
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from ctypes import *
from sklearn import datasets, metrics, model_selection
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
import pickle
from itertools import cycle
from sklearn.metrics import roc_curve, auc,roc_auc_score,confusion_matrix,precision_recall_curve,f1_score,average_precision_score,accuracy_score,classification_report
from scipy import interp
from sklearn.utils import shuffle

#Python Data operations functions
def missing_quantity(data,column):
    missing = pd.DataFrame(data[column].apply(lambda x: np.sum(x.isnull(), axis=0)))
    missing['count_missing'] = missing.iloc[:,0]
    missing['percentage_missing'] = (missing.iloc[:,0]/data.shape[0])*100
    missing.drop(missing.columns[0],axis='columns',inplace=True)
    return(missing)

	
def data_prep(transactions,currency_data,users):
#Class for variable transformation
    class cat_data_transform():
        def __init__(self):
            self.le = LabelEncoder()
            self.oe = OneHotEncoder(sparse = False)
        def onehot_encode(self,x):
            feature = self.le.fit_transform(x)
            self.oe = OneHotEncoder(sparse = False)
            return self.oe.fit_transform(feature.reshape(-1,1))
        def label_encode(self,x):
            return self.le.fit_transform(x)
        def inverse_enc(self,x):
            inv = list()
            for i in range(len(x)):
                inv.append(np.argmax(x[i]))
            inv=np.array(inv)    
            return self.le.inverse_transform(inv)

    #Feature Engineering- Transactions
    transactions.MERCHANT_CATEGORY[transactions.MERCHANT_CATEGORY.isna()] = 'marchant_category_not_known'
    transactions.MERCHANT_COUNTRY[transactions.MERCHANT_COUNTRY.isna()] = 'marchant_country_not_known'
    currency_type = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','CURRENCY']],index=['USER_ID'],columns='CURRENCY', aggfunc = np.count_nonzero)).reset_index(drop=True)
    currency_type['count_diff_currency']=[np.count_nonzero(np.array(currency_type.iloc[i,:])) for i in range(currency_type.shape[0])]
    currency_amount = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','CURRENCY','AMOUNT']],index=['USER_ID'],columns='CURRENCY',values = 'AMOUNT', aggfunc = np.sum)).reset_index(drop=True)
    def renaming_fun(x):
        return '_'.join([x,'amt'])
    currency_amount.columns = [renaming_fun(col) for col in list(currency_amount.columns)]
    state = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','STATE']],index=['USER_ID'],columns='STATE', aggfunc = np.count_nonzero))
    user_id = pd.DataFrame(state.index)
    state.reset_index(drop=True,inplace=True)
    MERCHANT_CATEGORY = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','MERCHANT_CATEGORY']],index=['USER_ID'],columns='MERCHANT_CATEGORY', aggfunc = np.count_nonzero)).reset_index(drop=True)
    MERCHANT_CATEGORY['count_diff_marchants']=[np.count_nonzero(np.array(MERCHANT_CATEGORY.iloc[i,:])) for i in range(MERCHANT_CATEGORY.shape[0])]
    MERCHANT_COUNTRY = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','MERCHANT_COUNTRY']],index=['USER_ID'],columns='MERCHANT_COUNTRY', aggfunc = np.count_nonzero)).reset_index(drop=True)
    def renaming_fun(x):
        return '_'.join(['country',x])
    MERCHANT_COUNTRY.columns = [renaming_fun(col) for col in list(MERCHANT_COUNTRY.columns)]
    MERCHANT_COUNTRY['count_diff_marchants_country']=[np.count_nonzero(np.array(MERCHANT_COUNTRY.iloc[i,:])) for i in range(MERCHANT_COUNTRY.shape[0])]
    ENTRY_METHOD = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','ENTRY_METHOD']],index=['USER_ID'],columns='ENTRY_METHOD', aggfunc = np.count_nonzero)).reset_index(drop=True)
    TYPE = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','TYPE']],index=['USER_ID'],columns='TYPE', aggfunc = np.count_nonzero)).reset_index(drop=True)
    TYPE['count_diff_type']=[np.count_nonzero(np.array(TYPE.iloc[i,:])) for i in range(TYPE.shape[0])]
    SOURCE = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','SOURCE']],index=['USER_ID'],columns='SOURCE', aggfunc = np.count_nonzero)).reset_index(drop=True)
    SOURCE['count_diff_SOURCE']=[np.count_nonzero(np.array(SOURCE.iloc[i,:])) for i in range(SOURCE.shape[0])]
    
    #Feature Engineering-Currency
    transactions = pd.merge(transactions,currency_data[['CCY','IS_CRYPTO']],how='left',left_on='CURRENCY',right_on='CCY')
    IS_CRYPTO = pd.DataFrame(pd.pivot_table(transactions[['USER_ID','IS_CRYPTO']],index=['USER_ID'],columns='IS_CRYPTO', aggfunc = np.count_nonzero)).reset_index(drop = True)
    IS_CRYPTO.columns = ['not_crypto','crypto']
    IS_CRYPTO['is_crypto'] = [1 if IS_CRYPTO.crypto[i] > 0 else 0 for i in range(IS_CRYPTO.shape[0])]
    user_level_data = pd.concat([user_id,currency_type,currency_amount,ENTRY_METHOD,TYPE,SOURCE,IS_CRYPTO,MERCHANT_CATEGORY,MERCHANT_COUNTRY],axis=1)
    user_level_data.fillna(0,inplace=True)
    
    #Feature Engineering-Users
    user_level_data = pd.merge(user_level_data,users,how='inner',left_on='USER_ID',right_on='ID')
    #User age during creation
    user_level_data['age'] = pd.Series(user_level_data['CREATED_DATE'].apply(lambda x: str(x).split('-')[0])).astype('int')-user_level_data['BIRTH_YEAR'].astype('int')

    #Flag for country and PHONE_COUNTRY different
    user_level_data['user_different_PHONE_COUNTRY'] = (np.array([str(user_level_data['COUNTRY'][i]) not in str(user_level_data['PHONE_COUNTRY'][i]) for i in range(len(user_level_data))])).astype('int32') 

    #TERMS_VERSION using(yearwise)
    user_level_data['TERMS_VERSION'] = pd.Series(user_level_data['TERMS_VERSION'].apply(lambda x: str(x).split('-')[0]))
    user_level_data['TERMS_VERSION'][user_level_data['TERMS_VERSION'] == 'nan'] = 'version_not_known'

    #Droping redundant columns
    user_id = user_level_data['USER_ID']
    user_level_data.drop(['PHONE_COUNTRY','STATE','CREATED_DATE', 'COUNTRY', 'BIRTH_YEAR','ID','USER_ID'], axis = 1, inplace=True)

    #Variable transformation 
    cat = cat_data_transform()
    user_level_data['IS_FRAUDSTER'] = cat.label_encode(user_level_data['IS_FRAUDSTER'])
    enc_kyc = pd.DataFrame(cat.onehot_encode(user_level_data['KYC']),columns=['FAILED','NONE','PASSED','PENDING'])
    enc_version = pd.DataFrame(cat.onehot_encode(user_level_data['TERMS_VERSION']),columns=['2017_vesion','2018_version','version_not_known'])
    user_level_data.drop(['KYC','TERMS_VERSION'],axis=1,inplace=True)
    final_user_level_data = pd.concat([user_level_data,enc_kyc,enc_version],axis=1)
    
    return final_user_level_data,user_id

def model_data_prep(final_user_level_data,user_id,features):
    train_X,valid_X,train_Y,valid_Y = train_test_split(final_user_level_data.iloc[:,final_user_level_data.columns!='IS_FRAUDSTER'],final_user_level_data.iloc[:,final_user_level_data.columns=='IS_FRAUDSTER'],test_size = 0.1, random_state = 20)
    train_X,valid_X=train_X[features],valid_X[features]
    user_train,user_valid = train_test_split(user_id,test_size = 0.1, random_state = 20)
    return train_X,valid_X,train_Y,valid_Y,user_train,user_valid
	
def model_train(train_X,train_Y,valid_X,valid_Y):
    bst_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bytree=0.8, gamma=0, learning_rate=0.05,
              max_delta_step=0, max_depth=4, min_child_weight=1.1, missing=None,
              n_estimators=1000, n_jobs=1, nthread=None,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=21, seed=20, silent=True,
              subsample=0.8)
    eval_set  = [(train_X,train_Y),(valid_X,valid_Y)]

    bst_mod = bst_model.fit(train_X,train_Y,eval_set=eval_set,early_stopping_rounds=20,verbose = False,
        eval_metric="auc")
    return bst_mod

def patrol(user,model_score,label):
    out_put = pd.concat([user],axis=1)
    out_put['IS_FRAUDSTER'] = label.values
    out_put['model_score'] = model_score
    out_put.sort_values('model_score',ascending=False,inplace=True)
    out_put['ventile'] = pd.qcut(out_put['model_score'].rank(method = 'first'),20,labels=range(20,0,-1))
    ventiles = pd.pivot_table(out_put,index=['ventile'],values=['model_score'],aggfunc=np.mean).reset_index(drop = True)
    ventiles['ventile'] = range(20,0,-1)
    ventiles['cnt_fraudstar'] = list(np.array(pd.pivot_table(out_put,index=['ventile'],values=['IS_FRAUDSTER'],aggfunc=np.sum).reset_index(drop = True)['IS_FRAUDSTER']))
    ventiles.sort_values('ventile',ascending=True,inplace=True)
    ventiles['cum_fraud_percent'] = np.cumsum(ventiles['cnt_fraudstar']/np.sum(ventiles['cnt_fraudstar']))
    ventiles['min_score'] = pd.pivot_table(out_put,index=['ventile'],values=['model_score'],aggfunc=np.min).reset_index(drop = True)
    ventiles['max_score'] = pd.pivot_table(out_put,index=['ventile'],values=['model_score'],aggfunc=np.max).reset_index(drop = True)
    v = np.array(out_put['model_score'])
    out_put['action'] = ['locked' if v[i] > ventiles['min_score'][17] else 'alart' if v[i] == ventiles['min_score'][17] else 'active' for i in range(len(v))]
    out_put = shuffle(out_put)
    print(out_put.head(5))
    return out_put
	
#Database operation functions

#Functions for creating table
def create_tables(config, connection):
    cur = connection.cursor()
    for table in config:
        name = table.get('name')
        schema = table.get('schema')
        ddl = f"""CREATE TABLE IF NOT EXISTS {name} ({schema})"""
        cur.execute(ddl)
    connection.commit()
	
#Functions for loading the tables with bulk data
def load_tables(config, connection,ip=True):
    """If ip==True it will load input data else it will load the model scored output data"""
    # iterate and load
    cur = connection.cursor()
    data_path_ip = "D:/data_science/revolut_challenge/data"     #Input data location
    data_path_op = "D:/data_science/revolut_challenge/deliverables/data/"  #Output data location
    for table in config:
        table_name = table.get('name')
        if ip == True:
            table_source = os.path.join(data_path_ip,(f"{table_name}.csv"))
        else:
            table_source = os.path.join(data_path_op,(f"{table_name}.csv"))
        with open(table_source, 'r') as f:
            next(f)
            cur.copy_expert(f"COPY {table_name} FROM STDIN CSV NULL AS ''", f)
        connection.commit()	
		
		