# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:56:42 2021

@author: Bharat Yadav
"""

# step 0 - import libraries
#------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

#-----------------------------------------------------------------------
# step 1 - read csv file
#-----------------------------------------------------------------------
bikes=pd.read_csv(r"C:\Users\Bharat Yadav\Desktop\bike_demand\hour.csv")

#------------------------------------------------------------------------
#step 2- prelim Analysis and feature selection
#------------------------------------------------------------------------
bikes_prep=bikes.copy()
bikes_prep=bikes_prep.drop(["index","date","casual","registered"],axis=1)

#basic check for null values
bikes_prep.isnull().sum()

#visualise the data using pandas histogram
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

#lecure 142 visualise the continous features vs demand
plt.subplot(2,2,1)
plt.title("Temperature vs Demand")
plt.scatter(bikes_prep["temp"],bikes_prep["demand"],s=2,c="g")

plt.subplot(2,2,2)
plt.title("aTemp vs Demand")
plt.scatter(bikes_prep["atemp"],bikes_prep["demand"],s=2,c="r")

plt.subplot(2,2,3)
plt.title("Humedity vs Demand")
plt.scatter(bikes_prep["humidity"],bikes_prep["demand"],s=2,c="b")

plt.subplot(2,2,4)
plt.title("windspeed vs Demand")
plt.scatter(bikes_prep["windspeed"],bikes_prep["demand"],s=2,c="c")

plt.tight_layout()

#plot the categorical value vs demand
#visualise the categorical features

colours=["g","r","b","m"]


plt.subplot(3,3,1)
plt.title("season vs demand")

cat_list=bikes_prep["season"].unique()
cat_average=bikes_prep.groupby("season").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(3,3,2)
plt.title("months vs demand")
cat_list=bikes_prep["month"].unique()
cat_average=bikes_prep.groupby("month").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)

plt.subplot(3,3,3)
plt.title("average demand per holiday")
cat_list=bikes_prep["holiday"].unique()
cat_average=bikes_prep.groupby("holiday").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)

plt.subplot(3,3,4)
plt.title("average demand per week day")
cat_list=bikes_prep["weekday"].unique()
cat_average=bikes_prep.groupby("weekday").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(3,3,5)
plt.title("average demand per year")
cat_list=bikes_prep["year"].unique()
cat_average=bikes_prep.groupby("year").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)

plt.subplot(3,3,6)
plt.title("average demand per weather")
cat_list=bikes_prep["weather"].unique()
cat_average=bikes_prep.groupby("weather").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)

plt.subplot(3,3,7)
plt.title("average demand per hour")
cat_list=bikes_prep["hour"].unique()
cat_average=bikes_prep.groupby("hour").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)

plt.subplot(3,3,8)
plt.title("average demand per workingday")
cat_list=bikes_prep["workingday"].unique()
cat_average=bikes_prep.groupby("workingday").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)

plt.tight_layout()
plt.show()

#--------------------------------------------------------------------
#check for outliers
#---------------------------------------------------------------------
bikes_prep["demand"].describe()
bikes_prep["demand"].quantile([0.05,0.1,0.15,0.9,0.95,0.99])

#--------------------------------------------------------------------
# step 4 check multiple linear regression assumption
#----------------------------------------------------------------------

#Lineraity using correlation cofficient matrix using corr
correlation=bikes_prep[["temp","atemp","humidity","windspeed","demand"]].corr()
 
bikes_prep=bikes_prep.drop(["atemp","weekday","year","workingday","windspeed"],axis=1)

#check autocorrelation
df=pd.to_numeric(bikes_prep["demand"],downcast="float")
plt.acorr(df,maxlags=12)

#--------------------------------------------------------------------
#sep 6 - Create/modify new features
#--------------------------------------------------------------------
#Log Normalise the feature "demand"
df1=bikes_prep["demand"]
df2=np.log(df1)

plt.figure()
df1.hist(rwidth=0.9,bins=20)

plt.figure()
df2.hist(rwidth=0.9,bins=20)

bikes_prep["deamand"]=np.log(bikes_prep["demand"])

# Autocorrelation in the demand column
t_1=bikes_prep["demand"].shift(+1).to_frame()
t_1.columns=["t-1"]

t_2=bikes_prep["demand"].shift(+2).to_frame()
t_2.columns=["t-2"]

t_3=bikes_prep["demand"].shift(+3).to_frame()
t_3.columns=["t-3"]

bikes_prep_lag=pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)

bikes_prep_lag=bikes_prep_lag.dropna()


#--------------------------------------------------------------------
# step 7- create dummy variable and drop first to avoid dummy variable 
#       trap using get_dummies
#--------------------------------------------------------------------

# - season,holiday,weather, month,hour
bikes_prep_lag.dtypes

bikes_prep_lag['season']=bikes_prep_lag['season'].astype('category')
bikes_prep_lag["holiday"]=bikes_prep_lag["holiday"].astype("category")
bikes_prep_lag["weather"]=bikes_prep_lag["weather"].astype("category")
bikes_prep_lag["month"]=bikes_prep_lag["month"].astype("category")

bikes_prep_lag["hour"]=bikes_prep_lag["hour"].astype("category")

bikes_prep_lag=pd.get_dummies(bikes_prep_lag,drop_first=True)
























