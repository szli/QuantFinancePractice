
# coding: utf-8

# ## Multifactor model for daily stock returns
# ##### szli@github, szli.code@gmail.com

# This exercise estimates BARRA factor model returns. More specifically,
# * The BARRA factor loading data and daily stock return data are given.
# * For simplicity, the analysis is limited to top around 600 stocks in market cap factor. 
# * Use cross sectional regressions to estimate factor returns, i.e., estimate factor returns $f_t$ given stock returns $R_t$ and factor loadings $B$ in the following: $R_t = Bf_t + \epsilon_t$, for each day $t$.
# 
# Here is a good reference for this topic: http://faculty.washington.edu/ezivot/research/factormodellecture_handout.pdf

# In[1]:

import pandas as pd
get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt
import numpy as np


# In[2]:

import sys
print(sys.version)
print(pd.version.version)
print(np.version.version)


# ### Load data, put factor loading and return data in good shape

# In[3]:

rawfactorcsv = pd.read_csv('./factor_loading.csv')
rawfactorcsv.rename(columns = {name: name.strip() for name in rawfactorcsv.columns}, inplace=True) #columns names have trailing spaces


# In[4]:

#Sort by CAPITALIZAION, pick first 1000 stocks, and set TICKER to be the index
sortedfactorcsv = rawfactorcsv.sort(columns='CAPITALIZATION', ascending=False)
sortedfactorcsv = sortedfactorcsv.set_index('TICKER', verify_integrity=True).iloc[:1000]


# In[5]:

#Only pick 12 factors, and clear the trailing spaces in the ticker
factors = sortedfactorcsv[['VOLTILTY','MOMENTUM','SIZE','SIZENONL','TRADEACT','GROWTH','EARNYLD','VALUE','EARNVAR','LEVERAGE', 'CURRSEN','YIELD']]
factors = factors.rename(index={sym:sym.strip() for sym in factors.index})


# In[6]:

factors


# In[7]:

stocks = factors.index
stocks


# Actually there are lots of symbols missing in price data, only 616 has data.... missing are on OTC markets...

# In[8]:

from os import listdir
pricefiles = listdir('./price_data_2013/')


# In[9]:

prices_date = {}
for pricefile in pricefiles:
    pricecsv = pd.read_csv('./price_data_2013/' + pricefile, sep = "\t")
    pricecsv.set_index('ticker', verify_integrity=True, inplace=True)    
    t = pricefile.split('.')[1]
    #only pick the stocks that are in 1000 stocks list, only use adjClose column, drop NaN rows (ticker in stocks but not in price files)
    prices_date[t] = pricecsv.loc[stocks].dropna()['adjClose']


# In[10]:

pricecsv.loc[stocks]


# In[11]:

pricesDF = pd.DataFrame(prices_date).T


# In[12]:

pricesDF


# In[13]:

#Columns with NaN prices
pd.isnull(pricesDF).any(0).nonzero()[0]


# In[14]:

pricesDF.iloc[:, 87]


# In[15]:

#Drop those NaN columns
pricesDF.dropna(1, inplace=True)


# In[16]:

returnDF = pricesDF.pct_change()


# In[17]:

returnDF


# In[18]:

returnDF.dropna(inplace=True) #Drop first row, all NaN


# In[19]:

#Only keep stocks that have price data in factor loading matrix, and reorder the matrix according to the ordering in return data
factors_trim = factors.loc[returnDF.columns]
factors_trim


# ### Cross-sectional regression:
# Note that we have heteroskedasticity in error terms, so first need to estimate the variance of the error term. We assume the error terms for each stock are uncorrelated, so the covariance matrix of the error terms will be diagnal. There are multiple ways to estimate error variance, I use two of them.
# 
# 1. Method 1: Perform OLS on every day, compute the residual $\hat{\epsilon}_t = R_t - B\hat{f_t}$ on every day, then use the sample variance of the residuals $\hat{\epsilon}_t$ across days as the estimate of the error variance, and perform Weighted Least Square using weights 1/variance_estimate
# 2. Method 2: Perform OLS on every day, then run the OLS linear regression below for every day
# $$log(\hat{\epsilon}^2_t) = g_{0t} + BG_t + e_t$$
# where $\hat{\epsilon}_t$ is the residual from OLS, $B$ is the factor loading matrix, used as the predictor of log variance (taking log to make sure estimated variance always non-negative). Then compute the fitted value of log variance: $B\hat{G}_t$, this gives the estimate of log variance on every day, take expontial to get the estimate of variance every day then average over days. Then, perform WLS using weights 1/variance_estimate. 
# 
# Comparing the estimated factor return using the two methods, they have similar trends over time, although the magnitudes are a bit different. 

# In[20]:

import numpy.linalg


# In[21]:

#returnDF is T by N, so need transpose, T = 20 days, N = 600+ stocks, P = 12 factors
#Do OLS for all days
OLS_factor_result = np.linalg.lstsq(factors_trim.values, returnDF.values.T) 


# In[22]:

OLS_factor_returns = OLS_factor_result[0]

OLS_factor_returns.shape #P by T


# In[23]:

#Note the dimensions: N-by-T -  N-by-P * P-by-T
OLS_residues = returnDF.values.T - factors_trim.values.dot(OLS_factor_returns)


# #### Method One: 

# In[24]:

var_est = np.var(OLS_residues, 1, ddof = 1)


# In[25]:

var_est


# In[26]:

WLS_weights = 1/var_est


# In[27]:

import statsmodels.api as sm


# In[28]:

WLS_results = {}
for row in returnDF.iterrows(): #row[1] is the Series, row[0] is the row name
    WLS_model = sm.WLS(row[1].values, factors_trim.values, weights = WLS_weights)
    WLS_results[row[0]] = WLS_model.fit()


# In[29]:

WLS_factor_return = {day: WLS_results[day].params for day in WLS_results}


# In[30]:

WLS_factor_returnDF = pd.DataFrame(data = WLS_factor_return, index = factors_trim.columns).T


# In[31]:

WLS_factor_returnDF


# In[32]:

for factor in WLS_factor_returnDF:
    plt.figure()
    plt.plot(WLS_factor_returnDF[factor])
    plt.title(factor)
    plt.xlabel("Days")
    plt.ylabel("Factor Return")
    


# In[33]:

for day in sorted(WLS_results):
    print(day)
    print(WLS_results[day].summary())


# #### Method Two: 

# In[34]:

log_resi_var = np.log(OLS_residues**2)


# In[35]:

residual_var_pred = sm.add_constant(factors_trim.values) # add constant term in regression
OLS_resi_var_result = np.linalg.lstsq(residual_var_pred, log_resi_var) 


# In[36]:

OLS_resi_var_result[0].shape #P * T


# In[37]:

#fitted value, take exp, then average over days
var_est2 = np.mean(np.exp(residual_var_pred.dot(OLS_resi_var_result[0])), axis = 1)


# In[38]:

var_est2.shape


# In[39]:

WLS_weights2 = 1/var_est2
WLS_results2 = {}
for row in returnDF.iterrows(): #row[1] is the Series, row[0] is the row name
    WLS_model2 = sm.WLS(row[1].values, factors_trim.values, weights = WLS_weights2)
    WLS_results2[row[0]] = WLS_model2.fit()
    
WLS_factor_return2 = {day: WLS_results2[day].params for day in WLS_results2}
WLS_factor_returnDF2 = pd.DataFrame(data = WLS_factor_return2, index = factors_trim.columns).T


# In[40]:

WLS_factor_returnDF2


# In[41]:

for factor in WLS_factor_returnDF2:
    plt.figure()
    plt.plot(WLS_factor_returnDF2[factor])
    plt.title(factor)
    plt.xlabel("Days")
    plt.ylabel("Factor Return")


# In[42]:

for day in sorted(WLS_results2):
    print(day)
    print(WLS_results2[day].summary())

