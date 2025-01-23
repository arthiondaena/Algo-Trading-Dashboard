import yfinance as yf
from backtesting import Backtest
import pandas as pd
import random

from strategies import SMC_test, SMC_ema, SMCStructure

def fetch(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.columns =df.columns.get_level_values(0)
    return df

def smc_plot_backtest(data, filename, swing_hl, **kwargs):
    bt = Backtest(data, SMC_test, **kwargs)
    bt.run(swing_window=swing_hl)
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
    bt = Backtest(data, SMC_test, **kwargs)
    results = bt.run(swing_window=swing_hl)
    bt.plot(filename='bokeh_graph.html', open_browser=False)
    return results

def smc_ema_backtest(data, ema1, ema2, closecross, **kwargs):
    bt = Backtest(data, SMC_ema, **kwargs)
    results = bt.run(ema1=ema1, ema2=ema2, close_on_crossover=closecross)
    bt.plot(filename='bokeh_graph.html', open_browser=False)
    return results

def smc_structure_backtest(data, swing_hl, **kwargs):
    bt = Backtest(data, SMCStructure, **kwargs)
    results = bt.run(swing_window=swing_hl)
    bt.plot(filename='bokeh_graph.html', open_browser=False)
    return results

def random_test(strategy: str, period: str, interval: str, no_of_stocks: int = 5, **kwargs):
    nifty50 = pd.read_csv("data/ind_nifty50list.csv")
    ticker_list = pd.read_csv("data/Ticker_List_NSE_India.csv")

    # Merging nifty50 and ticker_list dataframes to get 'YahooEquiv' column.
    nifty50 = nifty50.merge(ticker_list, "inner", left_on=['Symbol'], right_on=['SYMBOL'])

    # Generating random indices between 0 and len(nifty50).
    random_indices = random.sample(range(0, len(nifty50)), no_of_stocks)

    df = pd.DataFrame()

    for i in random_indices:
        # Fetching ohlc of random ticker_symbol.
        ticker_symbol = nifty50['YahooEquiv'].values[i]
        data = fetch(ticker_symbol, period, interval)

        if strategy == "Order Block":
            backtest_results = smc_backtest(data, kwargs['swing_hl'])
        elif strategy == "Order Block with EMA":
            backtest_results = smc_ema_backtest(data, kwargs['ema1'], kwargs['ema2'], kwargs['cross_close'])
        elif strategy == "Structure trading":
            backtest_results = smc_structure_backtest(data, kwargs['swing_hl'])
        else:
            raise Exception('Strategy not found')

        with open("bokeh_graph.html", 'r', encoding='utf-8') as f:
            plot = f.read()

        # Converting pd.Series to pd.Dataframe
        backtest_results = backtest_results.to_frame().transpose()

        backtest_results['stock'] = ticker_symbol

        # Reordering columns.
        # cols = df.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        cols = ['stock', 'Start', 'End', 'Return [%]', 'Equity Final [$]', 'Buy & Hold Return [%]', '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]']
        backtest_results = backtest_results[cols]

        df = pd.concat([df, backtest_results])

    df = df.sort_values(by=['Return [%]'], ascending=False)

    return df

def complete_test(strategy: str, period: str, interval: str, **kwargs):
    nifty50 = pd.read_csv("data/ind_nifty50list.csv")
    ticker_list = pd.read_csv("data/Ticker_List_NSE_India.csv")

    # Merging nifty50 and ticker_list dataframes to get 'YahooEquiv' column.
    nifty50 = nifty50.merge(ticker_list, "inner", left_on=['Symbol'], right_on=['SYMBOL'])

    df = pd.DataFrame()

    for i in range(len(nifty50)):
    # for i in range(5):

        # Fetching ohlc of random ticker_symbol.
        ticker_symbol = nifty50['YahooEquiv'].values[i]
        data = fetch(ticker_symbol, period, interval)

        if strategy == "Order Block":
            backtest_results = smc_backtest(data, kwargs['swing_hl'])
        elif strategy == "Order Block with EMA":
            backtest_results = smc_ema_backtest(data, kwargs['ema1'], kwargs['ema2'], kwargs['cross_close'])
        elif strategy == "Structure trading":
            backtest_results = smc_structure_backtest(data, kwargs['swing_hl'])
        else:
            raise Exception('Strategy not found')

        with open("bokeh_graph.html", 'r', encoding='utf-8') as f:
            plot = f.read()

        # Converting pd.Series to pd.Dataframe
        backtest_results = backtest_results.to_frame().transpose()

        backtest_results['stock'] = ticker_symbol
        backtest_results['plot'] = plot

        # Reordering columns.
        # cols = df.columns.tolist()
        # cols = cols[-1:] + cols[:-1]
        cols = ['stock', 'Start', 'End', 'Return [%]', 'Equity Final [$]', 'Buy & Hold Return [%]', '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'plot']
        backtest_results = backtest_results[cols]

        df = pd.concat([df, backtest_results])

    df['plot'] = df['plot'].astype(str)
    df = df.sort_values(by=['Return [%]'], ascending=False)

    return df


if __name__ == "__main__":
    # random_testing("")
    # data = fetch('RELIANCE.NS', period='1y', interval='15m')
    # df = yf.download('RELIANCE.NS', period='1yr', interval='15m')

    rt = complete_test("Order Block", '1mo', '15m', swing_hl=20)
    rt.to_excel('test/all_testing_1.xlsx', index=False)
    print(rt)