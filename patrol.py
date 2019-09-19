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
from functions import *

try:
    connection = pg.connect(
    host= 'localhost',password = 'Data@12345',
    port=5432,
    dbname='revolute_challenge',
    user='postgres')
    print('data base is connected')
except:
    print('Database not connected')

cur = connection.cursor()

os.chdir('D:/data_science/revolut_challenge/deliverables/data')
valid=pd.read_csv('D:/data_science/revolut_challenge/deliverables/data/validation_data.csv')
valid_X = valid.drop(['IS_FRAUDSTER','USER_ID'],axis=1)
valid_Y = valid[['IS_FRAUDSTER']]
user_valid = valid['USER_ID']

print('data read completed')
#Feature list retrieve from PG database

try:
    cur = connection.cursor()
    cur.execute("""SELECT * FROM feature_information""") 
    rows = cur.fetchall()
    print('data reading from database')
except:
    print('data not read')
	
feat_imp = list(rows[i][1] for i in range(len(rows)))	
valid_X = valid_X[feat_imp]
print('loading model to score on new data or validation data')
model = pickle.load(open("D:/data_science/revolut_challenge/deliverables/artifact/revolute_model.dat", "rb"))

print('Model Performance- not applicable to new or unlabelled data')
y_score = model.predict_proba(valid_X)
y_pred = model.predict(valid_X)
results = confusion_matrix(valid_Y.values, y_pred) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(valid_Y.values, y_pred) )
print('Report : ')
print(classification_report(valid_Y.values, y_pred)) 
auc = roc_auc_score(valid_Y.values, y_score[:,1])
f1 = f1_score(valid_Y.values, y_pred)
print('f1=%.3f auc=%.3f' % (f1,auc))

#User action
print('model implementing on users')
out_put = patrol(user=user_valid,model_score=y_score[:,1],label=valid_Y)
out_put.to_csv('D:/data_science/revolut_challenge/deliverables/data/output_score_action.csv',index=False)

#Loading output on database
#with open("D:/data_science/revolut_challenge/deliverables/misc/output_schemas.yaml") as schema_file:  #schema yaml for all Input tables
    #config = yaml.load(schema_file)
#create_tables(config, connection)
#load_tables(config, connection,ip=True)