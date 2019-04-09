# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:51:27 2019

@author: Kunal
"""


import csv
import numpy as np


with open('train.csv') as f:
    reader=csv.reader(f)
    data=[]
    for row in reader:
        data.append(row)
    
del row
data.remove(data[0])

data=np.asarray(data,dtype='float32')
X=data[:,0:-1]
Y=data[:,-1]

with open('test.csv') as f:
    reader=csv.reader(f)
    test=[]
    for row in reader:
        test.append(row)
    
del row
test.remove(test[0])

X_test=np.asarray(test,dtype='float32')
X_test=X_test[:,1:]

from sklearn.svm import SVC
svm=SVC(kernel='linear',probability=True,C=0.01)
svm.fit(X,Y)
svm.score(X,Y)

Y_res=svm.predict(X_test)

import pandas as pd

raw_data={'price_range':Y_res}

#raw_data={'price_range':Y_res}
df=pd.DataFrame(raw_data,columns=["price_range"])
#df=pd.DataFrame(raw_data,columns=["price_range"])
df.index=df.index+1
df.to_csv('output_SVM.csv',index_label='id',encoding='utf-8')