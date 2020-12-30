# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:29:39 2020

@author: WLEON
"""


import yfinance as yf
import random
import pandas as pd
import numpy as np
import json
import os

class Stock():
    def __init__(self,ticker,config):
        self.config = config
        self.ticker = ticker
        self.start = config.start
        self.end = config.end
        self.interval = config.interval
        
        self.raw_data = self._stock_download()
        
        # Clean date
        self._clean_data()
        
        # self._generate_feature()
        
    def _stock_download(self):
        return(yf.download(  # or pdr.get_data_yahoo(...
                # tickers list or string as well
                tickers = self.ticker,
        
                # use "period" instead of start/end
                # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                # (optional, default is '1mo')
                start = self.start,
                end = self.end,
        
                # fetch data by interval (including intraday if period < 60 days)
                # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                # (optional, default is '1d')
                interval = self.interval,
        
                # group by ticker (to access via data['SPY'])
                # (optional, default is 'column')
                group_by = 'ticker',
        
                # adjust all OHLC automatically
                # (optional, default is False)
                auto_adjust = True,
        
                # download pre/post regular market hours data
                # (optional, default is False)
                prepost = True,
        
                # use threads for mass downloading? (True/False/Integer)
                # (optional, default is True)
                threads = True,
        
                # proxy URL scheme use use when downloading?
                # (optional, default is None)
                proxy = None
            ))
    def _clean_data(self):
        df = self.raw_data.copy()
        
        if not isinstance(df.columns,pd.Index):
            self.ticker = df.columns.levels[0].values
        missing_row = df.apply(lambda x:sum(x.isna()),axis=1)
        
        drop_date = missing_row.loc[missing_row.values > int(len(self.ticker)/3*5)]
        
        if len(drop_date)>0:
            print(drop_date)
            df.drop(drop_date.index,axis=0,inplace=True)       
           
        if len(self.ticker) > 1:
            missing_column = df.apply(lambda x:sum(x.isna())).reset_index().sort_values(0)
            drop_ticker = missing_column.loc[missing_column[0]>1]['level_0'].unique()
            
            if len(drop_ticker)>0:
                df.drop(drop_ticker,axis=1,level=0,inplace=True)
                df.columns = df.columns.remove_unused_levels()
                self.ticker = df.columns.levels[0]
                
        self.clean_data=df
        
        if len(self.ticker) > 1:
            self.columns = pd.unique(self.clean_data.columns.get_level_values(1).values)
        else:
            self.columns = pd.unique(self.clean_data.columns)
            
            
            
class Config:
    start = '2015-01-01'
    end = '2020-12-01'
    interval = '1d'
    
config = Config()
stock = Stock(['SPY'],config)

from ta import add_momentum_ta,add_trend_ta
df = add_momentum_ta(stock.clean_data,high="High", low="Low", close="Close", volume="Volume", fillna=True)
df = add_trend_ta(df,high="High", low="Low", close="Close", fillna=True)
# df = add_all_ta_features(stock.clean_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
df.to_csv('data/SPY.csv')
