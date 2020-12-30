# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:58:19 2020

@author: WLEON
"""

from dataset import *
import pandas as pd
import torch


df = pd.read_csv(r'data/2324.TW.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

dataset = TimeSeriesDataset(df,[],'Close',10,1)
train_iter, test_iter = dataset.get_loaders(10)
