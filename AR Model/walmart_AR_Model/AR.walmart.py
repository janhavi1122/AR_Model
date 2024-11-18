# -- coding: utf-8 --
"""
Created on Mon Mar  4 15:21:30 2024

@author: Dnyaneshwari...

"""
#Time Series Prediction on Wallmart Dataset and Predict_new

import pandas as pd 
import numpy as np

walmart = pd.read_csv('E:/datascience/AR Model/walmart.csv')

#pre-processing 
walmart['t'] = np.arange(1,160)
walmart['t_square'] = walmart['t'] * walmart['t']
walmart['log_footfalls'] = np.log(walmart['Footfalls'])
walmart.columns
# ['Month', 'Footfalls', 't', 't_square', 'log_footfalls'] 
#In walmart data we have Jan-1991 to 0th column, we need only first 
#example - Jan from each cell 

p = walmart['Month'][0]
#before we will extract, let us create new column called 
#months to store extracted values 
p[0:3]

walmart['months'] = 0
#you can check the dataframe with months name with all values 0 
#the total records are 159 in walmart 
for i in range(159):
    p = walmart['Month'][i]
    walmart['months'][i] = p[0:3] 
    
month_dummies = pd.DataFrame(pd.get_dummies(walmart['months']))
#now let us concatenate these dummy values to dataframe 
walmart1 = pd.concat([walmart, month_dummies], axis=1)
#you can check the dataframe walmart1 

#Visualization - time Plot 
walmart1.Footfalls.plot()

#Data Partition 
Train = walmart1.head(147) 
Test = walmart1.tail(12)

#to change the index value in pandas dataframe 
#Test.set_index(np.arange(1,13)) 

############----Linear------############
import statsmodels.formula.api as smf 

linear_model = smf.ols('Footfalls ~ t' , data = Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))

rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear
#Out[27]: 209.92559265462643
####################Exponential#############################
Exp = smf.ols('log_footfalls ~ t' , data = Train).fit()
pred_Exp= pd.Series(Exp.predict(pd.DataFrame(Test['t'])))

rmse_Exp = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Exp))**2))
rmse_Exp
#Out[31]: 2062.950115467359
################---Quadratic----#########################
Quad = smf.ols('Footfalls ~ t + t_square' , data = Train).fit()
pred_Quad= pd.Series(Quad.predict(pd.DataFrame(Test[['t','t_square']])))

rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad))**2))
rmse_Quad
#Out[35]: 137.15462741356015
################################################################### 

#############Additive Seasonability ###############################3 
add_sea = smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_add_sea =  pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea
#Out[39]: 264.66439005687744
################Multiplicative Seasonality #################################### 

Mul_sea = smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_mul_sea =  pd.Series(Mul_sea.predict(Test))
rmse_mul_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_mul_sea))**2))
rmse_mul_sea
#Out[43]: 2062.9467323902477
#############-------Additive Seasonality Quadratic Trend --------------- 
add_sea_Quad = smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_mul_sea))**2))
rmse_add_sea_quad
# 2062.996088663918
############---------Multiplicative Seasonality Linear Trend-------------###
Mul_add_sea = smf.ols('log_footfalls ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_mul_add_sea =  pd.Series(Mul_add_sea.predict(Test))
rmse_mul_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_mul_add_sea))**2))
rmse_mul_add_sea
# 2062.9434993334708

#################-------------Consolidate-----------####################33 

data = {'MODEL':pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']),
        "RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
table_rmse = pd.DataFrame(data)
table_rmse
'''
               MODEL  RMSE_Values
0        rmse_linear   209.925593
1           rmse_Exp  2062.950115
2          rmse_Quad   137.154627
3       rmse_add_sea   264.664390
4       rmse_mul_sea  2062.996089
5  rmse_add_sea_quad  2062.996089
6   rmse_mul_add_sea  2062.943499
'''

#################---------Testing ################## 

predict_data = pd.read_excel('E:/datascience/AR Model/Copy of Predict_new(1).xlsx')

model_full = smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_model_full = pd.Series(add_sea.predict(Test[['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sep' , 'Oct' , 'Nov']]))
rmse_model_full = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_model_full))**2))
rmse_add_sea
#Out[58]: 264.66439005687744
pred_new = pd.Series(model_full.predict(predict_data))
pred_new
'''
0     2213.628216
1     2252.669534
2     2219.210851
3     2331.668836
4     2384.626820
5     2059.418138
6     2206.876122
7     2204.750773
8     2256.708757
9     2028.471300
10    1999.332467
11    2308.270556
dtype: float64

'''
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Autocorelation model AR
#calculate residuals from best model applied on full data
full_res=walmart1.Footfalls - model_full.predict(walmart1)
#ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
#to know whether there is autocorelation or not
tsa_plots.plot_acf(full_res,lags=12)

#ACT is an complete autocoreation fucntion gives values
#of auto coralation of any time series with its lagged valued

#PACF is a partial auto corelation function
#it finds corelations of present with lags of the residuals of the
tsa_plots.plot_pacf(full_res,lags=12)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#AR Model
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags = 1)
#model_ar = AutoReg(Train_res, lags = 12)
model_fit = model_ar.fit()
print('coefficients: %s' %model_fit.params)
'''
coefficients: const   -1.505706
y.L1     0.641099
dtype: float64
'''
pred_res=model_fit.predict(start=len(full_res),end=len(full_res)+len(predict_data)-1,dynamic=False)

pred_res.reset_index(drop=True,inplace=True)

#final model prediction using ASQT and AR(1) model
final_pred=pred_new + pred_res
final_pred
'''
final predicted residuals
0     2169.823067
1     2223.080383
2     2198.735566
3     2317.036441
4     2373.740298
5     2050.933092
6     2199.930660
7     2198.792337
8     2251.383103
9     2023.551322
10    1994.672566
11    2303.777391
dtype: float64

'''