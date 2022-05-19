#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:27:54 2021

@author: attila
"""


import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


atx_indices = pd.read_csv("atx.csv", sep = ";")
#print(atx_indices.columns)
#print(list(atx_indices["ISIN"]))



data = yf.download(list(atx_indices["ISIN"]), start="2020-12-01", end="2020-12-31")
#atx_companies = yf.Ticker(atx_indices["MIC"])
#atx_companies = atx_companies.history(start="2020-01-01", end="2020-12-31", interval="1d")
#aapl_historical["High"].plot()


data = data["Close"]
data = data.reset_index()
data = data.melt(id_vars = ["Date"], value_vars =  list(data.columns), var_name='Stock', value_name = 'Close_value')


engine = create_engine('postgresql://ds21m031:surf1234@dsc-inf.postgres.database.azure.com:5432/nyt_import')
sql = 'DROP TABLE IF EXISTS ds21_b1_jobs_stock;'
engine.execute(sql)
data.to_sql('ds21_b1_jobs_stock', engine)