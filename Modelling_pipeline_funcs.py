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


# In[ ]:


def feature_selection_plot(inp_df,inp_feature_count,inp_model_name):
    scaler=MinMaxScaler()
    inp_df.columns=['Features','Importance']
    importance_scaled=pd.DataFrame(scaler.fit_transform(inp_df[['Importance']].values),index=inp_df.index, columns=['Importance'])
    inp_df['Importance']=importance_scaled['Importance']
    inp_df.sort_values(by='Importance', inplace=True, ascending=True)
    df_plot=inp_df.tail(inp_feature_count).reset_index(drop=True)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=df_plot['Importance'],
                         y=df_plot['Features'],
                         orientation='h'))
    fig.update_traces(marker_color='steelblue')
    fig.update_xaxes(title='Importance')
    fig.update_yaxes(title='Features')
    fig.update_layout(title='Feature selection using {}'.format(inp_model_name))
    fig.show(config={'displaylogo':False})
    return inp_df


# In[ ]:


def cross_val_plot(cross_val_scores,model_name,metric_name):
    abs_error=[abs(ele) for ele in cross_val_scores]
    folds=["Fold {}".format(ele) for ele in range(1,len(cross_val_scores)+1)]
    fig=go.Figure([go.Bar(x=folds,y=abs_error,marker_color='steelblue')])
    title='{} fold cross-validation for {}'.format(len(cross_val_scores),model_name)
    fig.update_layout(title=title,
                     xaxis_title='Folds',
                     yaxis_title=metric_name, width=900,height=500)
    fig.show(config={'displaylogo':False})


# In[ ]:


def cross_valid_result(classifier,x_data_t,y_data_t,cv_k,params):
    skf=StratifiedKFold(n_splits=cv_k,shuffle=True,random_state=369)
    grid_search=GridSearchCV(classifier,param_grid=params,
                            cv=skf.split(x_data_t,y_data_t),
                            scoring='f1_macro',n_jobs=-1,verbose=3)
    model_opt=grid_search.fit(x_data_t,y_data_t)
    
    cv_res_df=pd.DataFrame(data=model_opt.cv_results_)
    cv_res_df.drop(['mean_fit_time','std_fit_time','mean_score_time','std_test_score','params'],axis=1,inplace=True)
    cv_res_df_col=list(cv_res_df.columns)
    cv_res_df_col.insert(0,cv_res_df_col.pop())
    cv_res_df=cv_res_df.reindex(columns=cv_res_df_col).sort_values(by='rank_test_score')
    
    opt_params=model_opt.best_params_
    classifier=classifier.set_params(**opt_params)
    final_model=classifier.fit(x_data_t,y_data_t)
    y_pred_t=final_model.predict(x_data_t)
    cv_scores=cross_val_score(classifier,x_data_t,y_data_t,cv=cv_k,scoring='f1_macro')
    clear_output()
    
    display(Markdown('__Cross validation for TRAIN dataset__'))
    display_data(cv_res_df.round(4))
    
    display(Markdown('__Best parameters__'))
    for k,v in opt_params.items():
        display(Markdown('__{}__ : {}'.format(k,v)))
        
    return final_model,cv_scores,y_pred_t,opt_params
    


# In[ ]:


def calculate_cm(cm):
    FP=cm.sum(axis=0)-np.diag(cm)
    FN=cm.sum(axis=1)-np.diag(cm)
    TP=np.diag(cm)
    TN=cm.sum()-(FP+FN+TP)
    
    FP=FP.astype(float)
    FN=FN.astype(float)
    TP=TP.astype(float)
    TN=TN.astype(float)
    
    return FP,FN,TP,TN


# In[ ]:


def model_metric_calculation(y_train,y_pred_train,x_valid,y_valid,model,model_name):
    classes=y_train.value_counts().index.to_list()
    
    acc_train=accuracy_score(y_train,y_pred_train)*100
    class_report_train=classification_report(y_train,y_pred_train,
                                            target_names=classes,output_dict=True)
    
    y_pred_valid=model.predict(x_valid)
    acc_valid=accuracy_score(y_valid,y_pred_valid)*100
    class_report_valid=classification_report(y_valid,y_pred_valid,
                                            target_names=classes,output_dict=True)
    
    kappa_train=cohen_kappa_score(y_train,y_pred_train)
    kappa_valid=cohen_kappa_score(y_valid,y_pred_valid)
    
    display(Markdown('***'))
    display(Markdown('__Overall statistics__'))
    display(Markdown('Accuracy on TRAIN DATA: {}%'.format(round(acc_train,4))))
    display(Markdown('Accuracy on VALIDATION DATA: {}%'.format(round(acc_valid,4))))
    
    display(Markdown('Kappa score on TRAIN DATA: {}%'.format(round(kappa_train,4))))
    display(Markdown('Kappa score on VALIDATION DATA: {}%'.format(round(kappa_valid,4))))
    display(Markdown('***'))

    cr_df_train=pd.DataFrame(class_report_train).T
    cr_df_valid=pd.DataFrame(class_report_valid).T
    
    train_cr_metric=cr_df_train[:len(classes)]
    valid_cr_metric=cr_df_valid[:len(classes)]
    
    train_cr_metric=train_cr_metric.iloc[:,:-1]
    valid_cr_metric=valid_cr_metric.iloc[:,:-1]
    
    cm_train=confusion_matrix(y_train,y_pred_train)
    cm_valid=confusion_matrix(y_valid,y_pred_valid)
    
    fp_t,fn_t,tp_t,tn_t=calculate_cm(cm_train)
    fp_v,fn_v,tp_v,tn_v=calculate_cm(cm_valid)
    
    train_spcfcty=tn_t/(tn_t+fp_t)
    valid_spcfcty=tn_v/(tn_v+fp_v)
    
    train_cr_metric['specificity']=list(train_spcfcty)
    valid_cr_metric['specificity']=list(valid_spcfcty)
    
    train_prev=(tp_t+fn_t)/(tp_t+fp_t+tn_t+fn_t)
    valid_prev=(tp_v+fn_v)/(tp_v+fp_v+tn_v+fn_v)
    
    train_cr_metric['prevalence']=list(train_prev)
    valid_cr_metric['prevalence']=list(valid_prev)
    
    train_b_acc=[sum(i)/2 for i in zip(list(train_cr_metric['recall']),list(train_spcfcty))]
    valid_b_acc=[sum(i)/2 for i in zip(list(valid_cr_metric['recall']),list(valid_spcfcty))]
    
    train_cr_metric['balanced accuracy']=train_b_acc
    valid_cr_metric['balanced accuracy']=valid_b_acc
    
    train_ppv=tp_t/(tp_t+fp_t)
    valid_ppv=tp_v/(tp_v+fp_v)
    
    train_cr_metric['PPV']=list(train_ppv)
    valid_cr_metric['PPV']=list(valid_ppv)
    
    train_npv=tn_t/(tn_t+fn_t)
    valid_npv=tn_v/(tn_v+fn_v)
    
    train_cr_metric['NPV']=list(train_npv)
    valid_cr_metric['NPV']=list(valid_npv)
    
    train_det_rate=tp_t/(tp_t+fp_t+tn_t+fn_t)
    valid_det_rate=tp_v/(tp_v+fp_v+tn_v+fn_v)
    
    train_cr_metric['detection rate']=list(train_det_rate)
    valid_cr_metric['detection rate']=list(valid_det_rate)
    
    train_det_prev=(tp_t+fp_t)/(tp_t+fp_t+tn_t+fn_t)
    valid_det_prev=(tp_v+fp_v)/(tp_v+fp_v+tn_v+fn_v)
    
    train_cr_metric['detection prevalence']=list(train_det_prev)
    valid_cr_metric['detection_prevalence']=list(valid_det_prev)
    
    train_cr_metric=train_cr_metric.T
    valid_cr_metric=valid_cr_metric.T
    
    train_cr_metric.insert(loc=0, column='metric', value=train_cr_metric.index)
    valid_cr_metric.insert(loc=0, column='metric', value=valid_cr_metric.index)
    
    display(Markdown('_Performance Metrics of **{}** for different classes: **TRAIN DATA**_'.format(model_name)))
    display_data(train_cr_metric.round(4))
    display(Markdown('_Performance Metrics of **{}** for different classes: **VALIDATION DATA**_'.format(model_name)))
    display_data(valid_cr_metric.round(4))
    
    train_avg_metric=cr_dt_train[len(classes)+1:].T
    valid_avg_metric=cr_dt_valid[len(classes)+1:].T
    
    train_avg_metric['micro_avg']=precision_recall_fscore_support(y_train,
                                                                 y_pred_train, average='micro')
    valid_avg_metric['macro_avg']=precision_recall_fscore_support(y_valid,
                                                                 y_pred_valid, average='macro')
    
    train_avg_metric=train_avg_metric.iloc[:-1,:]
    valid_avg_metric=valid_avg_metric.iloc[:-1,:]
    
    display(Markdown('_Performance Metrics of **{}** for different classes: **TRAIN DATA**_'.format(model_name)))
    display_data(train_avg_metric)
    display(Markdown('_Performance Metrics of **{}** for different classes: **VALIDATION DATA**_'.format(model_name)))
    display_data(valid_avg_metric)
    
    cm_fig=ff.create_annotated_heatmap(cm_train,x=classes,y=classes, colorscale='darkmint',showscale=True)
    cm_fig.update_layout(width=800, height=800, title='Confusion Matrix',xaxis_title="Predicted",yaxis_title="Reference")
    display(cm_fig)
    return train_avg_metric,valid_avg_metric,y_pred_valid,acc_train,acc_valid


# In[ ]:


def display_data(data_table,sel_col=None):
    data_table_series=[data_table[i] for i in data_table.columns]
    
    cell_color=[]
    n=len(data_table)
    for col in data_table.columns:
        if sel_col is None:
            cell_color.append(['mintcream']*n)
        else:
            if col!=sel_col:
                cell_color.append(['mintcream']*n)
            else:
                cell_color.append(['lightgreen']*n)
    fig=go.Figure(data=[go.Table(
        header=dict(values=list(data_table.columns),
                   fill_color='lightgreen',
                   align='center',
                   font=dict(color='white', size=15)
                   ),
        cells=dict(values=data_table_series,
                   fill_color='lightgreen',
                   align='center',
                   font=dict(color='black', size=11)
                   ))]) 
    
    if data_table.shape[0]==1:
        fig_ht=75
    elif data_table.shape[0]>=2 and data_table.shape[0]<=9:
        fig_ht=30*data_table.shape[0]
    else:
        fig_ht=300
        
    fig.update_layout(width=150*len(data_table.columns),
                     height=fig_ht,
                     margin=dict(l=0,r=0,b=0,t=0,pad=0))
    display(fig)
        


# In[ ]:


def target_var_set(df):
    
    if df[m.target_var].dtypes!=object:
        df[m.target_var]=df[m.target_var].astype('str')
        
    print(colored("Selected target variable is:",'magenta',attrs=['bold']),colored("{}",'blue',attrs=['bold']).format(m.target_var))
        


# In[ ]:


def cat_encoding(df):
    categorical_var=df.filter(m.encode_cols)
    one_hot_categorical_var=pd.get_dummies(categorical_var,columns=m.encode_cols
                                          ,drop_first=True)
    lr_data_df1=pd.concat([df,one_hot_categorical],axis=1,sort=False)
    lr_data_df1=lr_data_df1.drop(m.encode_cols,axis=1)
    
    df=lr_data_df1.copy()
    return df


# In[ ]:


def data_split(df,df_cp):
    split_percentage=m.split_amt
    cv_method=m.cv_input
    cv_k=m.cv_value
    
    df_transformed=df.copy()
    split_percentage=split_percentage/100
    X=df_transformed.loc[:,df.columns]
    y=df_transformed[m.target_var]
    X=X.drop(m.target_var,axis=1)
    
    x_train,x_valid,y_train,y_valid=train_test_split(X,y,test_size=split_percentage,
                                                    random_state=0,stratify=y)
    
    clear_output()
    print("Training-Validation Split Percentage:{split}".format(split=int((1-split_percentage)*100)))
    print("Total Observation:{obs}".format(obs=X.shape[0]))
    print("Training Observation:{train_obs}".format(train_obs=x_train.shape[0]))
    print("Validation Observation:{valid_obs}".format(valid_obs=x_valid.shape[0]))
    print("Selected k cross validation splits is:{cv_k}".format(cv_k=cv_k))
    
    il_valid=x_valid.index.tolist()
    x_valid_cp=df_cp.iloc[il_valid]
    return x_train,x_valid,y_train,y_valid,x_valid_cp,cv_k,cv_method


# In[ ]:


def linear_dependency(x_train):
    feature_space=x_train.loc[:,x_train.columns!=m.target_var]
    reduced_form,inds=sympy.Matrix(feature_space.values).rref()
    indep_features=[feature_space.columns.tolist()[i] for i in inds]
    collinear_features=[item for item in feature_space.columns if item not in set(indep_features)]
    
    if len(collinear_features)>0:
            print("\nFeatures causing linear dependency:{ld}".format(ld=" || ".join(collinear_features)))
            print("These variables will be removed which are causing linear dependency")
            x_train_processed=x_train[indep_features]
            print("Data contains {} rows and {} columns".format(x_train_processed.shape[0],x_train_processed_shape[1]))
            display(x_train_processed.dtypes)
    else:
        print("No linear dependent columns")
        x_train_processed=x_train.copy()
    
    return x_train_processed


# In[ ]:


def multi_collinear(x_train_processed):
    truncate_value=15
    exog_df=x_train_processed.loc[:,x_train_processed.columns!=m.target_var]
    exog_df=x_train_processed.copy()
    
    exog_df=add_constant(exog_df)
    vifs=pd.Series([1/(1.-OLS(exog_df[col].values,
                              exog_df.loc[:,exog_df.columns!=col].values).fit().rsquared) for col in exog_df],
                   index=exog_df.columns,
                   name='VIF')
    
    vifs=pd.DataFrame(vifs)
    vifs.drop('const',axis=0,inplace=True)
    vifs=vifs['VIF'].where(vifs['VIF']<=truncate_value,truncate_value)
    vifs=pd.DataFrame(vifs)
    vifs_df=vifs.sort_values(by=['VIF'],ascending=True)
    
    vifs_df['colors']=np.where(vifs_df.VIF<=5,'yellowgreen',
                                 np.where((vifs_df.VIF>5) & (vifs_df.VIF<=10),'steelblue','tomato'))
    
    Layout=go.Layout(title="VIF plot", xaxis=dict(title='VIF'),yaxis=dict(title='Features'))
    fig=go.Figure(go.Bar(x=vifs_df.VIF,y=vifs_df.index.tolist(),
                         orientation='h',marker_color=vifs_df['colors']),layout=Layout)
    fig.add_shape(type="line",x0=5,y0=0,x1=5,y1=len(vifs_df.index.tolist()),
                  line=dict(color="midnightblue",width=2,dash="dot"))
    fig.add_shape(type="line",x0=10,y0=0,x1=10,y1=len(vifs_df.index.tolist()),
                  line=dict(color="midnightblue",width=2,dash="dot"))
    
    fig.update_layout(width=1000,height=800)
    fig.show(config={'displaylogo':False})
    return vifs


def multi_col_vars(vifs):
    multicorr_vars=vifs['VIF'].loc[lambda x:x>=10].index.tolist()
    if len(multicorr_vars)>0:
        print("{}".format(len(multicorr_vars)),"Features causin high multi colinearity! {}".format('\n'.join(multicorr_vars)))
    else:
        print("No multi-colinearity in the feature space!")
        
    return multicorr_vars


def drop_multi_col_vars(multicorr_vars,x_train):
    col_to_drop=multicorr_vars
    
    if col_to_drop==[]:
        clear_output()
        print("No columns were dropped")
    else:
        clear_output()
        x_train.drop(col_to_drop,axis=1,inplace=True)
        display(Markdown('Successfully dropped:{}'.format(",".join(col_to_drop))))
        
        display(x_train.dtypes)
    return x_train


def variability(x_train,df):
    selected_features=[]
    selected_features_dict={}
    col_drop=[]
    zero_variance=(x_train.describe().loc['std']==0)
    zero_variance=zero_variance[zero_variance].index.tolist()
    if len(zero_variance)==0:
        print('No columns with 0 std dev.')
    else:
        display(pd.DataFrame(zero_varinace,columns=['Features']))
    #display(pd.DataFrame(df.describe().loc['std']).round(4))
        col_drop.extend(zero_variance)
    return selected_features,selected_features_dict,col_drop


# In[ ]:


def col_type_sep(df):
    cal_l=df.select_dtypes(exclude=["float64","int64"]).columns.tolist()
    num_l=df.select_dtypes(exclude=["object"]).columns.tolist()
    col_drop=[]
    if m.target_var in cat_l:
        cat_l.remove(m.target_var)
    else:
        num_l.remove(m.target_var)
        
    return cat_l,num_l,col_drop


def lin_dep(cat_l,num_l,df,col_drop):
    for i in cat_l:
        CategoryGroupLists=df.groupby(i)[m.target_var].apply(list)
        AnnovaResults=f_oneway(*CategoryGroupLists)
        if(AnnovaResults[1]>0.05):
            col_drop.append(i)
    for i in num_l:
        cr=df[i].corr(df[m.target_var])
        if(cr==0):
            col_drop.append(i)
            
    return col_drop


def variability_p(cat_l,num_l,col_drop):
    for i in num_l:
        if(df[i].describe().loc['std']==0):
            col_drop_append(i)
    return col_drop

def preprocess_df(col_drop,df):
    df=df[df.columns.difference(col_drop)]
    return df


# In[ ]:


def model_features(x_train_processed_balanced):
    feature_count=25
    
    selected_model_features={}
    
    X_train_selection=x_train_processed_balanced.copy()
    return X_train_selection,selected_model_features,feature_count


# In[ ]:


def feature_select_pls(x_train_processed_balanced,y_train_processed_balanced,selected_model_features,feature_count):
    feature_selection_pls='Y'
    if feature_selection_pls=='Y':
        X=x_train_processed_balanced.copy()
        y=y_train_processed_balanced.copy()
        
        clear_output()
        pls=PLSRegression(n_components=X.shape[1])
        le=LabelEncoder()
        y_=le.fit_transform(y)
        pls.fit(X,y_)
        
        minmax=MinMaxScaler()
        coeff=abs(pls.coef_)
        scaled_coeff=pd.DataFrame(minmax.fit_transform(coeff))*100
        
        df_plot_pls=pd.concat([pd.DataFrame(X.columns),scaled_coeff],axis=1)
        model_name="PLS"
        
        df_plot_pls=feature_selection_plot(df_plot_pls,feature_count,model_name)
        
        df_pls=df_plot_pls.sort_values(by="Importance", ascending=False).reset_index(drop=True)
        display(df_pls[['Features','Importance']].head(feature_count).round(4))
        selected_model_features.update({"PLS":df_pls.Features.tolist()[:feature_count]})
    else:
        clear_output()
        selected_model_features.pop("PLS",0)
        
    return selected_model_features


# In[ ]:


def final_feature_set(x_train_processed_balanced,selected_model_features):
    feature_selection_mode='U'
    add_features='N'
    col_add=["dummy1","dummy2"]
    
    if feature_selection_mode=='I':
        selected_features=list(reduce(set.intersection,(set(val) for val in selected_model_features.values())))
        if len(selected_features)==0:
            display(Markdown('__No common features__'))
    else:
        selected_features=list(reduce(set.union,(set(val) for val in selected_model_features.values())))
        
    if add_features=='Y':
        selected_features.extend(col_add)
        selected_features=list(set(selected_features))
        
    clear_output()
    if len(selected_features)==0:
        display(Markdown("No feature selected"))
        x_train_processed_balanced_subset=x_train_processed_balanced.copy()
    else:
        display(Markdown("__Features selected for modelling__"))
        display(pd.DataFrame(selected_features,columns=['Features']))
        final_features=selected_features
        x_train_processed_balanced_subset=x_train_processed_balanced[x_train_processed_balanced.columns.intersection(selected_features)]
    
    return x_train_processed_balanced_subset,selected_features,final_features


# In[ ]:


def model_var_set():
    ensemble_dict={'train':{},'valid':{}}
    selected_models={}
    
    acc_df=pd.DataFrame()
    
    precision_train=pd.DataFrame()
    precision_valid=pd.DataFrame()
    
    recall_train=pd.DataFrame()
    recall_valid=pd.DataFrame()
    
    f1_train=pd.DataFrame()
    f1_valid=pd.DataFrame()
    
    return ensemble_dict,selected_models,acc_df,precision_train,precision_valid,recall_train,recall_valid,f1_train,f1_valid

