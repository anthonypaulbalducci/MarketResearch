from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import requests

dat = yf.Ticker("MSFT")
dat.info
print(dat.history(period='1d').Open)

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)

if response.status_code == 200:
    content = response.text
    print(content)
else:
    print("Failed to retrieve the page.")

soup = BeautifulSoup(content, 'lxml')
table = soup.find('table')

