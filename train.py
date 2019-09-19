#Packages
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
from functions import *
 
print('packages installed')
os.chdir('D:/data_science/revolut_challenge/data')

#Database connection
try:
    connection = pg.connect(
    host= 'localhost',password = 'Data@12345',
    port=5432,
    dbname='revolute_challenge',
    user='postgres')
    print('data base is connected')
except:
    print('Database not connected')
	
	
os.chdir('D:/data_science/revolut_challenge/data')
users=pd.read_csv('users.csv')
transactions = pd.read_csv('transactions.csv')
fraudsters = pd.read_csv('fraudsters.csv')
currency_details = pd.read_csv('currency_details.csv')
countries = pd.read_csv('countries.csv')
print('data read completed')
print('preparing user level data')
user_level_data,user_id = data_prep(transactions=transactions,currency_data=currency_details,users=users)
print(user_level_data.shape)
print('checking missing value')
missing = missing_quantity(user_level_data,user_level_data.columns)
print('data has %d missing value' %(np.sum(missing['count_missing'])))

#Feature list retrieve from PG database

try:
    cur = connection.cursor()
    cur.execute("""SELECT * FROM feature_information""") 
    rows = cur.fetchall()
    print('data reading from database')
except:
    print('data not read')
	
feat_imp = list(rows[i][1] for i in range(len(rows)))	
train_X,valid_X,train_Y,valid_Y,user_train,user_valid = model_data_prep(final_user_level_data = user_level_data,user_id=user_id,features=feat_imp)
print('model data prepared')
print('model training..')
bst_mod = model_train(train_X,train_Y,valid_X,valid_Y)
print('saving model at artifacts')
pickle.dump(bst_mod, open("D:/data_science/revolut_challenge/artifact/revolute_model.dat", "wb"))