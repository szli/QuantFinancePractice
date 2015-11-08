
# coding: utf-8

# ## High Frequency Data Analysis Exercise
# ##### szli@github, szli.code@gmail.com

# This exercise analyze high frequency quote and trade data of four stocks. More specifically
# * Data cleanup and time stamp conversion
# * Autocorrelation of the tick-by-tick mid price return and trade price return
# * Correlation matrix of one minute return of the four stocks

# In[2]:

import pandas as pd
get_ipython().magic('pylab inline')
import matplotlib.pyplot as plt
import numpy as np
import numba


# In[3]:

import sys
print(sys.version)
print(pd.version.version)
print(np.version.version)


# ### 1. Data clean up

# In[4]:

rawcsv = pd.read_csv('./sample.taq.csv', header = None,  names = ['time', 'sym', 'type','d1','d2','d3','d4'],
                     nrows = 3139595, error_bad_lines=True, dtype={'d1': np.float64, 'd2': np.float64, 'd3':np.float64, 'd4':np.float64})


# In[5]:

rawcsv


# In[6]:

pd.isnull(rawcsv).any(0)


# No NaN in raw csv

# In[7]:

rawcsv['sym'].unique()


# starting from 3139596, four lines are exotic data, from 16:00:00 due to end of trading day, just ignore the remaining since we only care about 9:30 to 16:00
# 
# Caveat: Somehow the pandas CSV parser treats d1-d4 as string after a certain row, need to specify dtype to force casting. We need d2 to be floating point type because it could be ask price or trade size.

# In[8]:

rawcsv


# In[9]:

syms = rawcsv['sym'].unique() # ['SLB', 'JPM', 'NOV', 'WFC']


# In[10]:

quotes = {sym: pd.DataFrame for sym in syms}
trades = {sym: pd.DataFrame for sym in syms}


# In[11]:

grouped = rawcsv.groupby(['sym','type'])


# In[12]:

grouped.first() #This removes sym and type column in grouped.get_group


# In[13]:

for sym in syms:
    q = grouped.get_group((sym, 'Q')) 
    #only take between 9:30 and 16:00
    quotes[sym] = q[(q.time >= 93000) & (q.time <= 160000)] #.drop(['sym', 'type'], axis = 1)
    t = grouped.get_group((sym, 'T'))
    trades[sym] = t[(t.time >= 93000) & (t.time <= 160000)] #.drop(['sym', 'type'], axis = 1)
#quotes and trades contains raw price, size data for each symbol


# In[14]:

quotes['NOV']


# In[15]:

#Find the crossed quotes and remove them, print out the row number and time stamp of the crossed quote
for sym in syms:
    print(sym)
    print("Before dropping:", quotes[sym].shape)
    crossed_idx = quotes[sym].index[(quotes[sym].d1 > quotes[sym].d2).nonzero()[0]]
    print("Number of crossed quotes:", len(crossed_idx))
    print(quotes[sym].loc[crossed_idx])    
    quotes[sym] = quotes[sym].drop(crossed_idx)
    print("After dropping:", quotes[sym].shape)


# In[16]:

for sym in syms:
    print(sym)   
    crossed_idx = quotes[sym].index[(quotes[sym].d1 > quotes[sym].d2).nonzero()[0]]
    print(len(crossed_idx))


# In[17]:

#Take a look at five sigma outliers, note the index printed is the index (not label) in each individual data frame
for sym in syms:
    print(sym)
    print((np.abs(quotes[sym].d1 - quotes[sym].d1.mean() ) > 5 * quotes[sym].d1.std()).nonzero())
    print((np.abs(quotes[sym].d2 - quotes[sym].d2.mean() ) > 5 * quotes[sym].d2.std()).nonzero())
    print((np.abs(trades[sym].d1 - trades[sym].d1.mean() ) > 5 * trades[sym].d1.std()).nonzero())


# In[19]:

trades['WFC'].iloc[26173:26176] #focus on 26174


# In[20]:

trades['WFC'] = trades['WFC'].drop(trades['WFC'].index[26174])


# In[21]:

quotes['SLB'].iloc[68868:68871] #focus on 68869


# In[22]:

quotes['SLB'] = quotes['SLB'].drop(quotes['SLB'].index[68869])


# In[23]:

trades['JPM'].iloc[64698:64701] #focus on 64699


# In[24]:

trades['JPM'] = trades['JPM'].drop(trades['JPM'].index[64699])


# I also looked at 3-sigma outliers, but they are actually reasonable intraday moves. So we accept them.

# ### 2. Time Stamp Conversion

# I come up with an efficient way in Python using numba just-in-time compiler to convert customized timestamp to Pandas timestamp. It is able to convert 3 millions timestamp in abour 350 ms. 

# In[26]:

#Returns epoch time in us. The numba JIT compiler generates machine code, making it much faster than python
@numba.vectorize(['uint64(float64)'], target='cpu')
def toEpochUFunc(ts):
    ts = ts * 1000000 
    h = ts // 10000000000
    ts = ts % 10000000000
    m = ts // 100000000
    ts = ts % 100000000
    s = ts // 1000000
    us = ts % 1000000
    #assumes the date is Aug 21, 2015, since we do not know the date. Does not matter.
    return (1440115200 + h * 3600 + m * 60 + s) * 1000000 + us


# In[27]:

#How fast to convert all 3 millions timestamps to pandas DateTimeIndex
get_ipython().magic('time epons = toEpochUFunc(rawcsv.time.values) * 1000 #350 ms to convert to epoch ns')
get_ipython().magic("time pd.to_datetime(epons.astype('datetime64[ns]')) #10ms to create pandas DateTimeIndex")


# ### 3. Autocorrelation of quotes and trades

# In[28]:

#Note that we need return and price (for minutes price data)
quoteRet = {}
quotePrice = {}
for sym in syms:
    quotePrice[sym] = pd.DataFrame(index = pd.to_datetime(toEpochUFunc(quotes[sym].time.values) * 1000), 
                      data=(quotes[sym].d1.values + quotes[sym].d2.values)/2, columns=['Mid'])
    quoteRet[sym] = quotePrice[sym].pct_change().dropna()


# In[29]:

quotePrice['JPM'] #Just to give an example


# In[30]:

tradeRet = {}
for sym in syms:
    tradeRet[sym] = pd.DataFrame(index = pd.to_datetime(toEpochUFunc(trades[sym].time.values) * 1000), 
                                data=trades[sym].d1.values, columns=['Trd']).pct_change().dropna()
    


# In[31]:

#Autocorrelation of mid quote returns
lags = range(1,10)
for sym in syms:    
    ac = list(map(lambda lag: quoteRet[sym].Mid.autocorr(lag), lags))
    plt.figure()
    plt.plot(lags,ac)
    plt.title(sym)
    
             


# In[32]:

#Autocorrelation of trade returns
for sym in syms:
    ac = list(map(lambda lag: tradeRet[sym].Trd.autocorr(lag), lags))
    plt.figure()
    plt.plot(lags,ac)
    plt.title(sym)


# Trade return has stronger negative autocorrelation than mid quote returns. I suspect this is because trades happan between bid and ask prices, bouncing between bid and ask prices.
# 
# Quote return autocorrelation is a bit hard to explain to me... And I even suspect if it makes sense to compute quote by quote mid quote return autocorrelation, because lots of time the quote price does not change and return is just zero... 

# ### 4. Minutes return and correlation matrix

# In[33]:

#Here I sample mid quote to minutes. Note that we need to consider case when there is no data in that minute, here I just do simple things: just fill use previous minute price
min_ret = {}
for sym in syms:
    #This gives a Series, rather than a DF. dropna() is to drop the first time stamp NaN.
    min_ret[sym] = quotePrice[sym].resample('1min', how='median').fillna(method='ffill').pct_change().dropna()['Mid']    


# In[34]:

min_retDF = pd.DataFrame(data=min_ret)


# In[35]:

min_retDF


# In[36]:

#Correlation matrix of minute returns
min_corr = min_retDF.corr()
min_corr


# JPM and WFC both in finance industry, exhibits high correlation as expected. 
# NOV and SLV both in oil/NG industry, but the correlation is not as high as JPM and WFC.
