import yfinance as yf

def fetch(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.columns =df.columns.get_level_values(0)
    return df