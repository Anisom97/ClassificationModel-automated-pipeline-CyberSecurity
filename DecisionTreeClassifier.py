#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from IPython.display import HTML, display, Markdown, clear_output
import sys
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
#from cerberus.client import CerberusClient
#import awswrangler as wr
import numpy as mp
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
#import waterfall_ chart as wc
import plotly.express as px
import plotly.graph_objects as go
#import plotly.subplots as sp
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)
import holoviews as hv
hv.extension('bokeh')
from holoviews.plotting.util import process_cmap
import ipywidgets as widgets
import warnings
from scipy.stats import f_oneway
#importing the following libraries for modeling
#value- "Import Libraries for modeling"
             
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
             
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Lars
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
             
from sklearn_lvq import GlvqModel

from collections import defaultdict
import joblib
import pickle
             
from xgboost import XGBClassifier
             
from math import ceil,sqrt,pi,isnan

from statistics import mean
from pathlib import Path
from itertools import combinations, groupby, product
from io import StringIO

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LarsCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import sympy

# from imblearn.over_sampling import RandomOverSampler, SMOTE
# from imblearn.under_sampling import RandomUnderSampler, TomekLinks
# import smote_variants as sv

import pymssql
import pandas.io.sql as psql
from IPython import get_ipython
import warnings
import random
import logging

logger=logging.getLogger()
logger.setLevel(logging.CRITICAL)
random.seed(369)

from termcolor import colored
import re
from matplotlib import pyplot as plt
from matplotlib import gridspec,cm
from functools import reduce

import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import config as m

import plotly.io as pio
pio.renderers.default="notebook"

import Modelling_pipeline_funcs as mp


# In[ ]:


def DT_model(x_train,y_train,cv_k,cv_method):
    if cv_method=='Y':
        params=m.param_dt
        
        classifier=DecisionTreeClassifier(criterion='gini',splitter='best',random_state=369)
        dt_model,cv_scores_dt,y_pred_train_dt,yz=mp.cross_valid_result(classifier,x_train,y_train,cv_k,params)
        
    else:
        classifier=DecisionTreeClssifier(criterion='gini',splitter='best',max_depth=5,min_samples_split=3,min_sample_leaf=2)
        
        dt_model=classifier.fit(x_train,y_train)
        y_pred_train_dt=dt_model.predict(x_train)
        cv_scores_dt=[]
    return dt_model,cv_scores_dt,y_pred_train_dt


# In[ ]:


def cross_val_dt(cv_scores_dt):
    if cv_scores_dt==[]:
        pass
    else:
        mp.cross_val_plot(cv_scores_dt,"Decision Tree","F1-Score")


# In[ ]:


def DT_structure(dt_model,final_features):
    fig=plt.figure(figsize=(15,15))
    fig.suptitle('Decision tree plot on training data')
    _=plot_tree(dt_model,filled=True,max_depth=2,feature_names=final_features,class_names=dt_model.classes_)


# In[ ]:


def DT_metric(y_train,y_pred_train_dt,x_valid,y_valid,dt_model):
    train_avg_metric_dt,valid_avg_metric_dt,y_pred_valid_dt,acc_train_dt,acc_valid_dt=mp.model_metric_calculation(y_train,
                                                                                                              y_pred_train_dt,
                                                                                                              x_valid,
                                                                                                              y_valid,
                                                                                                              dt_model,
                                                                                                              'Decision Tree')
    return train_avg_metric_dt,valid_avg_metric_dt,y_pred_valid_dt,acc_train_dt, acc_valid_dt


# In[ ]:


def DT_store(y_pred_valid,x_valid_cp):
    pred_valid_dt=pd.DataFrame(y_pred_valid,columns=['pred_valid_dt'])
    x_valid_cp=x_valid_cp.reset_index().drop('index',axis=1)
    #x_valid_cp['xxx']=x_valid_cp['xxx'].astype(str)
    return x_valid_cp,pred_valid_dt

