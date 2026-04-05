"""
Top 50 S&P 500 components by market cap + SPY ETF.
"""
import json
from pathlib import Path

TARGET_TICKER = "SPY"

# Top 50 S&P 500 components by market capitalization (approximate, as of early 2026).
TOP50_TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "AMZN",   # Amazon
    "GOOGL",  # Alphabet Class A
    "META",   # Meta Platforms
    "BRK.B",  # Berkshire Hathaway
    "LLY",    # Eli Lilly
    "AVGO",   # Broadcom
    "JPM",    # JPMorgan Chase
    "TSLA",   # Tesla
    "UNH",    # UnitedHealth
    "XOM",    # Exxon Mobil
    "V",      # Visa
    "MA",     # Mastercard
    "PG",     # Procter & Gamble
    "JNJ",    # Johnson & Johnson
    "COST",   # Costco
    "HD",     # Home Depot
    "ABBV",   # AbbVie
    "WMT",    # Walmart
    "NFLX",   # Netflix
    "BAC",    # Bank of America
    "KO",     # Coca-Cola
    "CRM",    # Salesforce
    "MRK",    # Merck
    "CVX",    # Chevron
    "AMD",    # AMD
    "PEP",    # PepsiCo
    "ORCL",   # Oracle
    "TMO",    # Thermo Fisher
    "LIN",    # Linde
    "CSCO",   # Cisco
    "ACN",    # Accenture
    "ADBE",   # Adobe
    "MCD",    # McDonald's
    "ABT",    # Abbott Labs
    "WFC",    # Wells Fargo
    "DHR",    # Danaher
    "PM",     # Philip Morris
    "TXN",    # Texas Instruments
    "NEE",    # NextEra Energy
    "ISRG",   # Intuitive Surgical
    "QCOM",   # Qualcomm
    "INTU",   # Intuit
    "AMAT",   # Applied Materials
    "GE",     # GE Aerospace
    "CMCSA",  # Comcast
    "AMGN",   # Amgen
    "PFE",    # Pfizer
]


def get_all_tickers():
    """Return all tickers: Top 50 + SPY target."""
    tickers = sorted(set(TOP50_TICKERS + [TARGET_TICKER]))
    return tickers


def get_ticker_set():
    """Return a set of all tickers for fast lookup."""
    return set(get_all_tickers())


if __name__ == "__main__":
    tickers = get_all_tickers()
    print(f"Total tickers (Top 50 + SPY): {len(tickers)}")
    for t in tickers:
        print(f"  {t}")
