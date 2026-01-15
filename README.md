# MarketResearch
A collection of market research and analysis tools

S&P 500 ticket scraper:

Makes use of BeautifulSoup that scrapes the list of S&P 500 tickers from the relevant Wikipedia page and then queues YFinance to gather open, close, and volume data, finally saving to a Pandas dataframe and outputting as a CSV file. Default is set to collect one year's data.

Currency convert:

A utility to translate dollar values (up to $10,000) into their English equivalents with an accompanying reverse transpose function (i.e. 9010.05 -> nine thousand ten dollars and five cents -> 9010.05).

Tokenizer:

An experimental time series tokenizer.
