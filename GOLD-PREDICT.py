# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 14:48:46 2016

@author: DELL
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import  stats
from math import exp
#import arch
DATA=pd.read_csv("huangjin.csv",encoding='gbk')
data=[]
for i in DATA.index:
    data.append((DATA.ix[i,'high']+DATA.ix[i,'low'])/2)
df=pd.DataFrame(data,index=DATA['time']).dropna()
#plot(df) #黄金连续数据不平稳
data2log=np.log(df)
log=data2log.diff(1).dropna() #进行对数一阶拆分，进行收益率的估计
#plot(log) #拆分后平稳

fig = plt.figure(figsize=(10,5)) #做ACF与PACF图观察
ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax4=fig.add_subplot(224)
fig1 = sm.graphics.tsa.plot_acf(df,lags=30,ax=ax1)
fig2 = sm.graphics.tsa.plot_pacf(df,lags=30,ax=ax2)
fig3 = sm.graphics.tsa.plot_acf(log,lags=30,ax=ax3)
fig4 = sm.graphics.tsa.plot_pacf(log,lags=30,ax=ax4)
print fig1
print fig2
print fig3
print fig4

train = log[:-5]
test = log[-5:]
#print sm.tsa.arma_order_select_ic(train.values,max_ar=10,max_ma=10,ic='aic')['aic_min_order']
#确定ARMA(1,0)
order = (5,5) #根据ACF，PACF以及AIC选定（1，0），（4，4），（5，5）为参考
arma_mod = sm.tsa.ARMA(train.values,order).fit()
plt.figure(figsize=(15,5))
plt.plot(arma_mod.fittedvalues,label='fitted value')
plt.plot(train.values,label='real value')
plt.legend(loc=0)
#print arma_mod.aic
print len(train)
forecast=arma_mod.forecast(5)[0]
plt.figure(figsize=(10,4))
plt.plot(test.values,label='realValue')
plt.plot(forecast,label='predictValue')
plt.figure(figsize=(5,2))
plt.plot(forecast,label='predictValue')

rec = data2log[-6:].values[0]
recs=[]
recs.append(rec)
predicts=[]
for i in range(5):
    predicts.append(exp(recs[i]+forecast[i]))
    recs.append(recs[i]+forecast[i])
plt.figure(figsize=(5,5))
plt.plot(predicts,'r',label='predict value')
plt.plot(data[1213:1218],'blue',label='real value')    
plt.legend(loc=0)

