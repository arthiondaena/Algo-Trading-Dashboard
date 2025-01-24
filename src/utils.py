import yfinance as yf
from backtesting import Backtest
import pandas as pd
import os

from multiprocessing import Pool
from itertools import repeat
from functools import partial
from strategies import SMC_test, SMC_ema, SMCStructure
from src.colorer import get_logger, start_end_log

logger = get_logger()

@start_end_log
def fetch(symbol, period, interval):
    logger.info(f"Fetching {symbol} for interval {interval} and period {period}")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.columns =df.columns.get_level_values(0)
    return df

@start_end_log
def smc_backtest(data, filename, **kwargs):
    bt = Backtest(data, SMC_test, cash=kwargs['cash'], commission=kwargs['commission'])
    results = bt.run(swing_window=kwargs['swing_hl'])
    bt.plot(filename=filename, open_browser=False)
    return results

@start_end_log
def smc_ema_backtest(data, filename, **kwargs):
    bt = Backtest(data, SMC_ema, cash=kwargs['cash'], commission=kwargs['commission'])
    results = bt.run(swing_window=kwargs['swing_hl'], ema1=kwargs['ema1'], ema2=kwargs['ema2'], close_on_crossover=kwargs['cross_close'])
    bt.plot(filename=filename, open_browser=False)
    return results

@start_end_log
def smc_structure_backtest(data, filename, **kwargs):
    bt = Backtest(data, SMCStructure, cash=kwargs['cash'], commission=kwargs['commission'])
    results = bt.run(swing_window=kwargs['swing_hl'])
    bt.plot(filename=filename, open_browser=False)
    return results

@start_end_log
def run_strategy(ticker_symbol, strategy, period, interval, **kwargs):
    logger.info(f'Running {strategy} for {ticker_symbol}')
    # Fetching ohlc of random ticker_symbol.
    retries = 3
    for i in range(retries):
        try:
            data = fetch(ticker_symbol, period, interval)
        except:
            raise Exception(f"{ticker_symbol} data fetch failed")

        if len(data) == 0:
            if i < retries - 1:
                print(f"Attempt{i + 1}: {ticker_symbol} ohlc is empty")
            else:
                raise Exception(f"{ticker_symbol} ohlc is empty")
        else:
            break

    filename = f'{ticker_symbol}.html'

    if strategy == "Order Block":
        backtest_results = smc_backtest(data, filename, **kwargs)
    elif strategy == "Order Block with EMA":
        backtest_results = smc_ema_backtest(data, filename, **kwargs)
    elif strategy == "Structure trading":
        backtest_results = smc_structure_backtest(data, filename, **kwargs)
    else:
        raise Exception('Strategy not found')

    with open(filename, 'r', encoding='utf-8') as f:
        plot = f.read()

    os.remove(filename)

    # Converting pd.Series to pd.Dataframe
    backtest_results = backtest_results.to_frame().transpose()

    backtest_results['stock'] = ticker_symbol
    backtest_results['plot'] = plot

    # Reordering columns.
    cols = ['stock', 'Start', 'End', 'Return [%]', 'Equity Final [$]', 'Buy & Hold Return [%]', '# Trades',
            'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'plot']
    backtest_results = backtest_results[cols]

    backtest_results = backtest_results.rename(columns = {'Equity Final [$]': 'Equity Final [₹]'})

    return backtest_results

@start_end_log
def complete_test(strategy: str, period: str, interval: str, multiprocess=True, **kwargs):
    nifty50 = pd.read_csv("data/ind_nifty50list.csv")
    ticker_list = pd.read_csv("data/Ticker_List_NSE_India.csv")

    # Merging nifty50 and ticker_list dataframes to get 'YahooEquiv' column.
    nifty50 = nifty50.merge(ticker_list, "inner", left_on=['Symbol'], right_on=['SYMBOL'])

    if multiprocess:
        with Pool() as p:
            result = p.starmap(partial(run_strategy, **kwargs), zip(nifty50['YahooEquiv'].values, repeat(strategy), repeat(period), repeat(interval)))
    else:
        result = [run_strategy(nifty50['YahooEquiv'].values[i], strategy, period, interval, **kwargs) for i in range(len(nifty50))]

    df = pd.concat(result)

    df['plot'] = df['plot'].astype(str)
    df = df.sort_values(by=['Return [%]'], ascending=False)

    return df.reset_index().drop(columns=['index'])


if __name__ == "__main__":
    # random_testing("")
    # data = fetch('RELIANCE.NS', period='1y', interval='15m')
    # df = yf.download('RELIANCE.NS', period='1yr', interval='15m')

    rt = complete_test("Order Block", '1mo', '15m', swing_hl=20)
    rt.to_excel('test/all_testing_2.xlsx', index=False)
    print(rt)