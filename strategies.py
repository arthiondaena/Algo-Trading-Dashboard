from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy
from indicators import SMC
from data_fetcher import fetch
import pandas as pd

class SMC_test(Strategy):
    def init(self):
        super().init()

        self.smc_b = self.I(self.smc_buy, self.data.df)
        self.smc_s = self.I(self.smc_sell, self.data.df)


    def next(self):
        price = self.data.Close[-1]
        current_time = self.data.index[-1]

        if self.smc_b[-1] == 1:
            self.buy(sl=.95 * price, tp=1.05 * price)
        if self.smc_s[-1] == -1:
            self.sell(tp=.95 * price, sl=1.05 * price)

        # Additionally, set aggressive stop-loss on trades that have been open
        # for more than two days
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, self.data.Low[-1])
                else:
                    trade.sl = min(trade.sl, self.data.High[-1])


    def smc_buy(self, data):
        return SMC(data).backtest_buy_signal()

    def smc_sell(self, data):
        return SMC(data).backtest_sell_signal()

class SMC_ema(SignalStrategy, TrailingStrategy):
    pass

def plot_backtest(data, strategy, filename, **kwargs):
    bt = Backtest(data, strategy, **kwargs)
    bt.run()
    return bt.plot(filename=filename, open_browser=False)

if __name__ == "__main__":
    data = fetch('ICICIBANK.NS', period='1mo', interval='15m')
    bt = Backtest(data, SMC_test, commission=.002)

    bt.run()
    bt.plot()