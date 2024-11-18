# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:01:42 2024

@author: santo
"""

import pandas as pd 
import numpy as np
import seaborn as sns
##############################################################

import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

Walmart = pd.read_csv('E:/datascience/AR Model/walmart.csv')
#Data Partition
Train=Walmart.head(147)
Test=Walmart.tail(12)
'''
In order to use this model will need to first find out 
p representation number of auto corelation terms lgas 
q presents number of moving average terms lags 
d represents number of non seasonal differences 
to find the value of pdq we use auto correlation ACF and 
parallel auto correlation that is PACF of lags
'''
tsa_plots.plot_acf(Walmart.Footfalls,lags=12)

tsa_plots.plot_pacf(Walmart.Footfalls,lags=12)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model1=ARIMA(Train.Footfalls,order=(3,1,5))
res1=model1.fit()
print(res1.summary())
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
start_index=len(Train)
end_index=start_index+11
forecast_test=res1.predict(start=start_index,end=end_index)
print(forecast_test)
'''
147    1947.791114
148    1868.147422
149    1863.482010
150    2001.406261
151    1971.248508
152    1818.479078
153    1842.283830
154    1990.957830
155    1964.802040
156    1816.217287
157    1842.871826
158    1990.517267
Name: predicted_mean, dtype: float64

'''
#evaluate forecast
rmse_test=sqrt(mean_squared_error(Test.Footfalls,forecast_test))
print("Total rmse ",rmse_test)
#Total rmse  180.0737831021779
#plot forecast against actual outcomes
pyplot.plot(Test.Footfalls)
pyplot.plot(forecast_test,color='red')
pyplot.show()

#Auto ARIMA
#pip install pmdarima --user

import pmdarima as pm
ar_model=pm.auto_arima(Train.Footfalls,)
