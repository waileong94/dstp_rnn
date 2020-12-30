# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:07:31 2020

@author: WLEON
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:29:39 2020

@author: WLEON
"""

import random
import pandas as pd
import numpy as np
import json
import os

df = pd.read_csv(r'data/crypto/raw/Binance_BTCUSDT_1h.csv',skiprows=1)
df['unix'] = np.where(df['unix']>1e12,df['unix']/1000,df['unix'])
df['unix'] = pd.to_datetime(df['unix'],unit='s')
df.rename(columns = {'unix':'Date','Volume BTC':'Volume','open':'Open','high':'High','low':'Low','close':'Close'},inplace=True)
df.drop(columns=['date','tradecount','symbol','Volume USDT'],inplace=True)
df.set_index('Date',inplace=True)
df = df.sort_values('Date')

from ta import add_momentum_ta,add_trend_ta
df = add_momentum_ta(df,high="High", low="Low", close="Close", volume="Volume", fillna=True)
df = add_trend_ta(df,high="High", low="Low", close="Close", fillna=True)
# df = add_all_ta_features(stock.clean_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

df = df.loc['2020':]
df.to_csv('data/BTCUSDT.csv')
