#!/usr/bin/env python
# coding: utf-8

# In[ ]:


target_var='Result'

split_amt=20

cv_input='Y'
cv_value=5

param_dt={'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                  'min_samples_split':[1,2,3,4,5,6,7,8,9,10],
                  'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],
                  'max_features':[2,3,4,5,6,'sqrt','log2'],
                  'class_weight':[{0:1,1:5,-1:4},{0:1,1:3,-1:3},'balanced']
         }