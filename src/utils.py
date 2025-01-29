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
    results = bt.run(swing_window=kwargs['swing_hl'], ema1=kwargs['ema1'], ema2=kwargs['ema2'], close_on_crossover=kwargs['close_on_crossover'])
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
    default_kwargs = {'swing_hl': 10, 'ema1': 9, 'ema2':21, 'close_on_crossover': False, 'cash': 10000, 'commission': 0}
    kwargs = default_kwargs | kwargs
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

    backtest_results['Stock'] = ticker_symbol
    backtest_results['plot'] = plot
    backtest_results['Sector'] = yf.Ticker(ticker_symbol).info.get('sectorKey')
    backtest_results['Return [%]'] = backtest_results['Return [%]'].apply(lambda x: round(x, 2))

    # Reordering columns.
    cols = ['Stock', 'Sector', 'Start', 'End', 'Return [%]', 'Equity Final [$]', 'Buy & Hold Return [%]', '# Trades',
            'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'plot']
    backtest_results = backtest_results[cols]

    backtest_results = backtest_results.rename(columns = {'Equity Final [$]': 'Equity Final [₹]'})

    return backtest_results

@start_end_log
def complete_test(stock_list: str, strategy: str, period: str, interval: str, multiprocess: bool, **kwargs):
    stock_list_map = {'Nifty 50': 'data/ind_nifty50list.csv', 'Nifty Next 50': 'data/ind_niftynext50list.csv', 'Nifty 100': 'data/ind_nifty100list.csv', 'Nifty 200': 'data/ind_nifty200list.csv'}
    nifty_stocks = pd.read_csv(stock_list_map[stock_list])
    nifty_stocks.columns = [x.upper() for x in nifty_stocks.columns]
    logger.info(f"stock list columns: {nifty_stocks.columns}")
    ticker_list = pd.read_csv("data/Ticker_List_NSE_India.csv")

    # Merging nifty50 and ticker_list dataframes to get 'YahooEquiv' column.
    nifty_stocks = nifty_stocks.merge(ticker_list, "inner", 'SYMBOL')

    if multiprocess:
        with Pool() as p:
            result = p.starmap(partial(run_strategy, **kwargs), zip(nifty_stocks['YahooEquiv'].values, repeat(strategy), repeat(period), repeat(interval)))
    else:
        result = [run_strategy(nifty_stocks['YahooEquiv'].values[i], strategy, period, interval, **kwargs) for i in range(len(nifty_stocks))]

    df = pd.concat(result)

    df['plot'] = df['plot'].astype(str)
    df = df.sort_values(by=['Return [%]'], ascending=False)

    return df.reset_index().drop(columns=['index'])

def categorize_df(df: pd.DataFrame, col: str, sort_col: str | None = None):
    categorized = df.groupby(col, sort=False)
    mapping = {}
    for name, group in categorized:
        mapping[name] = group
        # print(f"{name} mean: ", group[sort_col].mean())
    # print(sorted(mapping.values(), key = lambda item: item[sort_col].mean(), reverse=True))
    if sort_col:
        mapping = dict([('all', df)]+sorted(mapping.items(), key = lambda item: item[1][sort_col].mean(), reverse=True))

    for category, df in mapping.items():
        mapping[category] = df.sort_values(by=[sort_col], ascending=False)
    # print(mapping)
    return mapping

if __name__ == "__main__":
    # pass
    # random_testing("")
    # data = fetch('RELIANCE.NS', period='1y', interval='15m')
    # df = yf.download('RELIANCE.NS', period='1yr', interval='15m')
    # rt.to_excel('test/all_testing_2.xlsx', index=False)
    #
    # print(rt)

    data = pd.read_csv(r"C:\Users\Dinesh\Downloads\Documents\2025-01-26T12-37_export.csv")
    data = data[data['Select']]
    print(data)
    mapping = categorize_df(data, 'Sector', 'Return [%]')
    print(mapping)