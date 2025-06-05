import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def scrape_sp500_tickers(): 
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)

    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    tickers = []

    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if cols:
            ticker = cols[0].text.strip().replace(".", "-")  # yfinance uses "-" instead of "."
            tickers.append(ticker)

    return tickers


def fetch_stock_data(tickers, period='1y'):
    data_list = []

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                continue

            hist = hist[['Open', 'Close', 'Volume']].copy()
            hist['Ticker'] = ticker
            hist['Date'] = hist.index
            data_list.append(hist)
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
            continue

    if data_list:
        all_data = pd.concat(data_list)
        all_data.reset_index(drop=True, inplace=True)
        return all_data
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    tickers = scrape_sp500_tickers()
    stock_data = fetch_stock_data(tickers)

    if not stock_data.empty:
        stock_data.to_csv("sp500_stock_data.csv", index=False)
        print("Data saved to sp500_stock_data.csv")
    else:
        print("No data fetched.")
