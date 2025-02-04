from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy, TrailingStrategy
from indicators import SMC, EMA
import pandas as pd
import numpy as np

from src.colorer import get_logger

logger = get_logger()

class SMC_test(Strategy):
    swing_window = 10
    def init(self):
        super().init()

        # Setting smc buy and sell indicators.
        self.smc_b = self.I(self.smc_buy, data=self.data.df, swing_hl=self.swing_window)
        self.smc_s = self.I(self.smc_sell, data=self.data.df, swing_hl=self.swing_window)

    def next(self):
        price = self.data.Close[-1]
        current_time = self.data.index[-1]

        # If buy signal, set target 5% above price and stoploss 5% below price.
        if self.smc_b[-1] == 1:
            self.buy(sl=.95 * price, tp=1.05 * price)
        # If sell signal, set targe 5% below price and stoploss 5% above price.
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

    def smc_buy(self, data, swing_hl):
        return SMC(data, swing_hl).backtest_buy_signal_ob()

    def smc_sell(self, data, swing_hl):
        return SMC(data, swing_hl).backtest_sell_signal_ob()


class SMC_ema(SignalStrategy, TrailingStrategy):
    swing_window = 10
    ema1 = 9
    ema2 = 21
    close_on_crossover = False

    def init(self):
        super().init()

        # Setting smc buy and sell indicators.
        self.smc_b = self.I(self.smc_buy, data=self.data.df, swing_hl=self.swing_window)
        self.smc_s = self.I(self.smc_sell, data=self.data.df, swing_hl=self.swing_window)

        close = self.data.Close

        # Setting up EMAs.
        self.ma1 = self.I(EMA, close, self.ema1)
        self.ma2 = self.I(EMA, close, self.ema2)


    def next(self):
        price = self.data.Close[-1]
        current_time = self.data.index[-1]

        # If buy signal and short moving average is above long moving average.
        if self.smc_b[-1] == 1 and self.ma1 > self.ma2:
            self.buy(sl=.95 * price, tp=1.05 * price)
        # If sell signal and short moving average is below long moving average.
        if self.smc_s[-1] == -1 and self.ma1 < self.ma2:
            self.sell(tp=.95 * price, sl=1.05 * price)

        # Additionally, set aggressive stop-loss on trades that have been open
        # for more than two days
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, self.data.Low[-1])
                else:
                    trade.sl = min(trade.sl, self.data.High[-1])

        # Close the trade if there is a moving average crossover in opposite direction
        if self.close_on_crossover:
            for trade in self.trades:
                if trade.is_long and self.ma1 < self.ma2:
                    trade.close()
                if trade.is_short and self.ma1 > self.ma2:
                    trade.close()

    def smc_buy(self, data, swing_hl):
        return SMC(data, swing_hl).backtest_buy_signal_ob()

    def smc_sell(self, data, swing_hl):
        return SMC(data, swing_hl).backtest_sell_signal_ob()


class SMCStructure(TrailingStrategy):
    swing_window = 20

    def init(self):
        super().init()
        self.smc_b = self.I(self.smc_buy, data=self.data.df, swing_hl=self.swing_window)
        self.smc_s = self.I(self.smc_sell, data=self.data.df, swing_hl=self.swing_window)
        self.set_trailing_sl(2)
        # self.swing = self.I(self.nearest_swing, data=self.data.df, swing_hl)

    def next(self):
        price = self.data.Close[-1]
        current_time = self.data.index[-1]

        if self.smc_b[-1] == 1:
            nearest = self.nearest_swing(self.data.df, self.swing_window)
            target = price + ((price - nearest)* .414)
            stoploss = price - (target-price)
            # print(f"buy: {current_time}, {price}, {nearest}, {target}, {stoploss}")
            try:
                self.buy(sl=stoploss, tp=target)
            except:
                logger.warning(f'Buying failed at {price} with {stoploss=} and {target=}')
        if self.smc_s[-1] == 1:
            nearest = self.nearest_swing(self.data.df, self.swing_window)
            if nearest > price:
                target = price - ((nearest - price) * .414)
                stoploss = price + (price - target)
                # print(f"sell: {current_time}, {price}, {nearest}, {target}, {stoploss}")
                try:
                    self.sell(sl=stoploss, tp=target, limit=float(price))
                except:
                    logger.warning(f'Selling failed at {price} with {stoploss=} and {target=}')

        # Additionally, set aggressive stop-loss on trades that have been open
        # for more than two days
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, self.data.Low[-1])
                else:
                    trade.sl = min(trade.sl, self.data.High[-1])

    def smc_buy(self, data, swing_hl):
        return SMC(data, swing_hl).backtest_buy_signal_structure()

    def smc_sell(self, data, swing_hl):
        return SMC(data, swing_hl).backtest_sell_signal_structure()

    def nearest_swing(self, data, swing_hl):
        # Get swing high/low nearest to current price.
        swings = SMC(data, swing_hl).swing_hl
        swings = swings[~np.isnan(swings['Level'])]
        return swings['Level'].iloc[-2]

strategies = {'Order Block': SMC_test, 'Order Block with EMA': SMC_ema , 'Structure trading': SMCStructure}

if __name__ == "__main__":
    from src.utils import fetch
    # data = fetch('ICICIBANK.NS', period='1mo', interval='15m')
    data = fetch('RELIANCE.NS', period='1mo', interval='15m')
    # data = fetch('AXISBANK.NS', period='1mo', interval='15m')
    # bt = Backtest(data, SMC_ema, commission=.002)
    # bt.run(ema1 = 9, ema2 = 21, close_on_crossover=True)
    bt = Backtest(data, SMCStructure, commission = .002, trade_on_close=True)
    print(bt.run())

    # bt.plot()