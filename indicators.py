import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SMC:
    def __init__(self, data, swing_hl_window_sz=10):
        self.data = data
        self.data['Date'] = self.data.index.to_series()
        self.swing_hl_window_sz = swing_hl_window_sz
        self.order_blocks = self.order_block()

    def backtest_buy_signal(self):
        bull_ob = self.order_blocks[(self.order_blocks['OB']==1) & (self.order_blocks['MitigatedIndex']!=0)]
        arr = np.zeros(len(self.data))
        arr[bull_ob['MitigatedIndex'].apply(lambda x: int(x))] = 1
        return arr

    def backtest_sell_signal(self):
        bear_ob = self.order_blocks[(self.order_blocks['OB'] == -1) & (self.order_blocks['MitigatedIndex'] != 0)]
        arr = np.zeros(len(self.data))
        arr[bear_ob['MitigatedIndex'].apply(lambda x: int(x))] = -1
        return arr

    def swing_highs_lows(self, window_size):
        l = self.data['Low'].reset_index(drop=True)
        h = self.data['High'].reset_index(drop=True)
        swing_highs = (h.rolling(window_size, center=True).max() / h == 1.)
        swing_lows = (l.rolling(window_size, center=True).min() / l == 1.)
        return pd.DataFrame({'Date':self.data.index.to_series(), 'highs':swing_highs.values, 'lows':swing_lows.values})

    def fvg(self):
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.

        parameters:

        returns:
        FVG = 1 if bullish fair value gap, -1 if bearish fair value gap
        Top = the top of the fair value gap
        Bottom = the bottom of the fair value gap
        MitigatedIndex = the index of the candle that mitigated the fair value gap
        """

        fvg = np.where(
            (
                    (self.data["High"].shift(1) < self.data["Low"].shift(-1))
                    & (self.data["Close"] > self.data["Open"])
            )
            | (
                    (self.data["Low"].shift(1) > self.data["High"].shift(-1))
                    & (self.data["Close"] < self.data["Open"])
            ),
            np.where(self.data["Close"] > self.data["Open"], 1, -1),
            np.nan,
        )

        top = np.where(
            ~np.isnan(fvg),
            np.where(
                self.data["Close"] > self.data["Open"],
                self.data["Low"].shift(-1),
                self.data["Low"].shift(1),
            ),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                self.data["Close"] > self.data["Open"],
                self.data["High"].shift(1),
                self.data["High"].shift(-1),
            ),
            np.nan,
        )

        mitigated_index = np.zeros(len(self.data), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(self.data), dtype=np.bool_)
            if fvg[i] == 1:
                mask = self.data["Low"][i + 2:] <= top[i]
            elif fvg[i] == -1:
                mask = self.data["High"][i + 2:] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        return pd.concat(
            [
                pd.Series(fvg.flatten(), name="FVG"),
                pd.Series(top.flatten(), name="Top"),
                pd.Series(bottom.flatten(), name="Bottom"),
                pd.Series(mitigated_index.flatten(), name="MitigatedIndex"),
            ],
            axis=1,
        )

    def order_block(self, imb_perc=.1, join_consecutive=True):
        hl = self.swing_highs_lows(self.swing_hl_window_sz)

        ob = np.where(
            (
                    ((self.data["High"]*((100+imb_perc)/100)) < self.data["Low"].shift(-2))
                    & ((hl['lows']==True) | (hl['lows'].shift(1)==True))
            )
            | (
                    (self.data["Low"] > (self.data["High"].shift(-2)*((100+imb_perc)/100)))
                    & ((hl['highs']==True) | (hl['highs'].shift(1)==True))
            ),
            np.where(((hl['lows']==True) | (hl['lows'].shift(1)==True)), 1, -1),
            np.nan,
        )

        # print(ob)

        top = np.where(
            ~np.isnan(ob),
            np.where(
                self.data["Close"] > self.data["Open"],
                self.data["Low"].shift(-2),
                self.data["Low"],
            ),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(ob),
            np.where(
                self.data["Close"] > self.data["Open"],
                self.data["High"],
                self.data["High"].shift(-2),
            ),
            np.nan,
        )

        # if join_consecutive:
        #     for i in range(len(ob) - 1):
        #         if ob[i] == ob[i + 1]:
        #             top[i + 1] = max(top[i], top[i + 1])
        #             bottom[i + 1] = min(bottom[i], bottom[i + 1])
        #             ob[i] = top[i] = bottom[i] = np.nan

        mitigated_index = np.zeros(len(self.data), dtype=np.int32)
        for i in np.where(~np.isnan(ob))[0]:
            mask = np.zeros(len(self.data), dtype=np.bool_)
            if ob[i] == 1:
                mask = self.data["Low"][i + 3:] <= top[i]
            elif ob[i] == -1:
                mask = self.data["High"][i + 3:] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 3
                mitigated_index[i] = int(j)
        ob = ob.flatten()
        mitigated_index1 = np.where(np.isnan(ob), np.nan, mitigated_index)

        return pd.concat(
            [
                pd.Series(ob.flatten(), name="OB"),
                pd.Series(top.flatten(), name="Top"),
                pd.Series(bottom.flatten(), name="Bottom"),
                pd.Series(mitigated_index1.flatten(), name="MitigatedIndex"),
            ],
            axis=1,
        ).dropna(subset=['OB'])

    def plot(self, swing_hl=True, show=True):
        fig = make_subplots(1, 1)

        # plot the candle stick graph
        fig.add_trace(go.Candlestick(x=self.data.index.to_series(),
                        open=self.data['Open'],
                        high=self.data['High'],
                        low=self.data['Low'],
                        close=self.data['Close'],
                        name='ohlc'))

        # grab first and last observations from df.date and make a continuous date range from that
        dt_all = pd.date_range(start=self.data['Date'].iloc[0], end=self.data['Date'].iloc[-1], freq='5min')

        # check which dates from your source that also accur in the continuous date range
        dt_obs = [d.strftime("%Y-%m-%d %H:%M:%S") for d in self.data['Date']]

        # isolate missing timestamps
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d %H:%M:%S").tolist() if not d in dt_obs]

        # adjust xaxis for rangebreaks
        fig.update_xaxes(rangebreaks=[dict(dvalue=5 * 60 * 1000, values=dt_breaks)])

        print(self.order_blocks.head())
        print(self.order_blocks.index.to_list())

        ob_df = self.data.iloc[self.order_blocks.index.to_list()]
        # print(ob_df)

        fig.add_trace(go.Scatter(
                        x=ob_df['Date'],
                        y=ob_df['Low'],
                        name="Order Block",
                        mode='markers',
                        marker_symbol='diamond-dot',
                        marker_size=13,
                        marker_line_width=2,
                        # offsetgroup=0,
                       ))

        if swing_hl:
            hl = self.swing_highs_lows(self.swing_hl_window_sz)
            h = hl[(hl['highs']==True)]
            l = hl[hl['lows']==True]
            # print(h)
            # exit(0)
            fig.add_trace(go.Scatter(
                x=h['Date'],
                y=self.data[self.data.Date.isin(h['Date'])]['High']*(100.1/100),
                mode='markers',
                marker_symbol="triangle-up-dot",
                marker_size=10,
                name='Swing High',
                # offsetgroup=2,
            ))
            fig.add_trace(go.Scatter(
                x=l['Date'],
                y=self.data[self.data.Date.isin(l['Date'])]['Low']*(99.9/100),
                mode='markers',
                marker_symbol="triangle-down-dot",
                marker_size=10,
                name='Swing Low',
                marker_color='red',
                # offsetgroup=2,
            ))

        fig.update_layout(xaxis_rangeslider_visible=False)
        if show:
            fig.show()
        return fig


if __name__ == "__main__":
    from data_fetcher import fetch
    data = fetch('ICICIBANK.NS', period='1mo', interval='15m')
    # data = fetch('RELIANCE.NS', period='1mo', interval='15m')

    # print(SMC(data).backtest_buy_signal())
    SMC(data).plot()