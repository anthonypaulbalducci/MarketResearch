# MarketResearch
A collection of market research and analysis tools

S&P 500 ticket scraper.

Makes use of BeautifulSoup that scrapse the list of S&P 500 tickers from the relevant Wikipedia page and then queues YFinance to gather open, close, and volume data, finally saving to a Pandas dataframe and outputting as a CSV file. Default is set to collect one year's data.
