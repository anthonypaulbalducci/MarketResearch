from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

dat = yf.Ticker("MSFT")
dat.info
print(dat.history(period='1d'))
