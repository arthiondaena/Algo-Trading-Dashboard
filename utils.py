import yfinance as yf
from backtesting import Backtest

from strategies import SMC_test, SMC_ema, SMCStructure

def fetch(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.columns =df.columns.get_level_values(0)
    return df

def smc_plot_backtest(data, filename, swing_hl, **kwargs):
    bt = Backtest(data, SMC_test, **kwargs)
    bt.run(swing_hl=swing_hl)
    return bt.plot(filename=filename, open_browser=False)

def smc_ema_plot_backtest(data, filename, ema1, ema2, closecross, **kwargs):
    bt = Backtest(data, SMC_ema, **kwargs)
    bt.run(ema1=ema1, ema2=ema2, close_on_crossover=closecross)
    return bt.plot(filename=filename, open_browser=False)

def smc_structure_plot_backtest(data, filename, swing_hl, **kwargs):
    bt = Backtest(data, SMCStructure, **kwargs)
    bt.run(swing_window=swing_hl)
    return bt.plot(filename=filename, open_browser=False)

def smc_backtest(data, swing_hl, **kwargs):
    return Backtest(data, SMC_test, **kwargs).run(swing_hl=swing_hl)

def smc_ema_backtest(data, ema1, ema2, closecross, **kwargs):
    return Backtest(data, SMC_ema, **kwargs).run(ema1=ema1, ema2=ema2, close_on_crossover=closecross)

def smc_structure_backtest(data, swing_hl, **kwargs):
    return Backtest(data, SMCStructure, **kwargs).run(swing_hl=swing_hl)

if __name__ == "__main__":
    # data = fetch('RELIANCE.NS', period='1y', interval='15m')
    df = yf.download('RELIANCE.NS', period='1yr', interval='15m')