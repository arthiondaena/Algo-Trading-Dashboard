import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SMC:
    def __init__(self, data, swing_hl_window_sz=10):
        """
        Smart Money Concept
        :param data:
            Should contain Open, High, Low, Close columns and 'Date' as index
        :type data: pd.DataFrame
        :param swing_hl_window_sz: {int}
            CHoCH Detection Period
        """
        self.data = data
        self.data['Date'] = self.data.index.to_series()
        self.swing_hl_window_sz = swing_hl_window_sz
        self.order_blocks = self.order_block()
        self.swing_hl = self.swing_highs_lows_v2(self.swing_hl_window_sz)
        self.structure_map = self.bos_choch(self.swing_hl)

    def backtest_buy_signal_ob(self):
        """
        :return:
            Getting buy signals from order blocks mitigation index
        :rtype: np.ndarray
        """
        bull_ob = self.order_blocks[(self.order_blocks['OB']==1) & (self.order_blocks['MitigatedIndex']!=0)]
        arr = np.zeros(len(self.data))
        arr[bull_ob['MitigatedIndex'].apply(lambda x: int(x))] = 1
        return arr

    def backtest_sell_signal_ob(self):
        """
        :return:
            Getting sell signals from order blocks mitigation index
        :rtype: np.ndarray
        """
        bear_ob = self.order_blocks[(self.order_blocks['OB'] == -1) & (self.order_blocks['MitigatedIndex'] != 0)]
        arr = np.zeros(len(self.data))
        arr[bear_ob['MitigatedIndex'].apply(lambda x: int(x))] = -1
        return arr

    def backtest_buy_signal_structure(self):
        """
        :return:
            Getting buy signals from bullish structure broken index
        :rtype: np.ndarray
        """
        bull_struct = self.structure_map[(self.structure_map['BOS'] == 1) | (self.structure_map['CHOCH'] == 1)]
        arr = np.zeros(len(self.data))
        arr[bull_struct['BrokenIndex'].apply(lambda x: int(x))] = 1
        return arr

    def backtest_sell_signal_structure(self):
        """
        :return:
            Getting buy signals from bullish structure broken index
        :rtype: np.ndarray
        """
        bull_struct = self.structure_map[(self.structure_map['BOS'] == -1) | (self.structure_map['CHOCH'] == -1)]
        arr = np.zeros(len(self.data))
        arr[bull_struct['BrokenIndex'].apply(lambda x: int(x))] = 1
        return arr

    def swing_highs_lows(self, window_size):
        """
        Basic version of swing highs and lows. Suitable for finding swing order blocks.
        :param window_size:
            Window size for searching swing highs and lows
        :type window_size: int
        :return:
            DataFrame with Date, highs(bool), lows(bool) columns
        :rtype: pd.DataFrame
        """
        l = self.data['Low'].reset_index(drop=True)
        h = self.data['High'].reset_index(drop=True)
        swing_highs = (h.rolling(window_size, center=True).max() / h == 1.)
        swing_lows = (l.rolling(window_size, center=True).min() / l == 1.)
        return pd.DataFrame({'Date':self.data.index.to_series(), 'highs':swing_highs.values, 'lows':swing_lows.values})

    def swing_highs_lows_v2(self, window_size):
        """
        Updated version of swing_highs_lows function. Suitable for BOS and CHoCH.
        :param window_size:
            Window size for searching swing highs and lows.
        :type window_size: int
        :return:
            DataFrame with HighLow(1 for bull, -1 for bear), Level columns.
        :rtype: pd.DataFrame
        """
        l = self.data['Low'][::-1].reset_index(drop=True)
        h = self.data['High'][::-1].reset_index(drop=True)
        swing_highs = (h.rolling(window_size, min_periods=1).max() / h == 1.)[::-1]
        swing_lows = (l.rolling(window_size, min_periods=1).min() / l == 1.)[::-1]

        swing_highs.reset_index(drop=True, inplace=True)
        swing_lows.reset_index(drop=True, inplace=True)

        swings = np.where((swing_highs | swing_lows), np.where(swing_highs, 1, -1), 0)

        state = 1
        for i in range(1, swings.shape[0]):
            if swings[i] == state or swings[i] == 0:
                swings[i] = 0
            else:
                state *= -1

        swing_highs_lows = np.where(swings==0, np.nan, swings)

        # Get positions of swing_highs_lows where elements are not np.nan
        pos = np.where(~np.isnan(swing_highs_lows))[0]

        if len(pos) > 0:
            if swing_highs_lows[pos[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[pos[0]] == -1:
                swing_highs_lows[0] = 1
            if swing_highs_lows[pos[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[pos[-1]] == 1:
                swing_highs_lows[-1] = -1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, self.data.High, self.data.Low),
            np.nan,
        )

        return pd.concat(
            [
                pd.Series(swing_highs_lows, name="HighLow"),
                pd.Series(level, name="Level"),
            ],
            axis=1,
        )

    def bos_choch(self, swing_highs_lows):
        """
        Break of Structure and Change of Character
        :param swing_highs_lows: A DataFrame which contains swing highs and lows.
            Format should be same as swing_highs_lows_v2() function.
        :type swing_highs_lows: pd.DataFrame
        :return: A DataFrame with BOS(1 for bear, -1 for bull),
            CHOCH(1 for bear, -1 for bull), Level, BrokenIndex as columns.
        :rtype: pd.DataFrame
        """
        level_order = []
        highs_lows_order = []

        bos = np.zeros(len(self.data), dtype=np.int32)
        choch = np.zeros(len(self.data), dtype=np.int32)
        level = np.zeros(len(self.data), dtype=np.float32)

        last_positions = []

        for i in range(len(swing_highs_lows["HighLow"])):
            if not np.isnan(swing_highs_lows["HighLow"][i]):
                level_order.append(swing_highs_lows["Level"][i])
                highs_lows_order.append(swing_highs_lows["HighLow"][i])
                if len(level_order) >= 4:
                    # bullish bos
                    #                   -1
                    #        -3 __BOS__ / \
                    #        / \       /   \
                    #       /   \     /
                    #  \   /     \   /
                    #   \ /        -2
                    #   -4
                    bos[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-4]
                                < level_order[-2]
                                < level_order[-3]
                                < level_order[-1]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] !=0 else 0
                    )

                    # bearish bos
                    #   -4
                    #   / \        -2
                    #  /   \       / \
                    #       \     /   \
                    #        \   /     \
                    #         \ /__BOS__\   /
                    #         -3         \ /
                    #                     -1
                    bos[last_positions[-2]] = (
                        -1
                        if(
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-4]
                                > level_order[-2]
                                > level_order[-3]
                                > level_order[-1]
                            )
                        )
                        else bos[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bullish CHoCH
                    #                     -1
                    #        -3 __CHoCH__ / \
                    #        / \         /   \
                    #       /   \       /
                    #  \   /     \     /
                    #   \ /       \   /
                    #    -4        \ /
                    #               -2
                    choch[last_positions[-2]] = (
                        1
                        if (
                                np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                                and np.all(
                            level_order[-1]
                            > level_order[-3]
                            > level_order[-4]
                            > level_order[-2]
                        )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                    # bearish CHoCH
                    #              -2
                    #   -4         / \
                    #   / \       /   \
                    #  /   \     /     \
                    #       \   /       \
                    #        \ /         \
                    #         -3__CHoCH__ \   /
                    #                      \ /
                    #                       -1
                    choch[last_positions[-2]] = (
                        -1
                        if (
                                np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                                and np.all(
                            level_order[-1]
                            < level_order[-3]
                            < level_order[-4]
                            < level_order[-2]
                        )
                        )
                        else choch[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                last_positions.append(i)

        broken = np.zeros(len(self.data), dtype=np.int32)
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(self.data), dtype=np.bool_)
            # if the bos is 1 then check if the candles high has gone above the level
            if bos[i] == 1 or choch[i] == 1:
                mask = self.data.Close[i + 2:] > level[i]
            # if the bos is -1 then check if the candles low has gone below the level
            elif bos[i] == -1 or choch[i] == -1:
                mask = self.data.Close[i + 2:] < level[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j
                # if there are any unbroken bos or CHoCH that started before this one and ended after this one then remove them
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    if k < i and broken[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0

        # remove the ones that aren't broken
        for i in np.where(
                np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0

        # replace all the 0s with np.nan
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level = np.where(level != 0, level, np.nan)
        broken = np.where(broken != 0, broken, np.nan)

        bos = pd.Series(bos, name="BOS")
        choch = pd.Series(choch, name="CHOCH")
        level = pd.Series(level, name="Level")
        broken = pd.Series(broken, name="BrokenIndex")

        return pd.concat([bos, choch, level, broken], axis=1)

    def fvg(self):
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.

        :return:\
        FVG = 1 if bullish fair value gap, -1 if bearish fair value gap
        Top = the top of the fair value gap
        Bottom = the bottom of the fair value gap
        MitigatedIndex = the index of the candle that mitigated the fair value gap
        :rtype: pd.DataFrame
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
        """
        OB - Order Block
        Order block is the presence of a chunk of market orders that results in a sudden rise or fall in the market

        :return:\
        OB = 1 if bullish order block, -1 if bearish order block
        Top = the top of the order block
        Bottom = the bottom of the order block
        MitigatedIndex = the index of the candle that mitigated the order block
        :rtype: pd.DataFrame
        """
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

    def plot(self, order_blocks=False, swing_hl=False, swing_hl_v2=False, structure=False, show=True):
        """
        :param order_blocks:
        :param swing_hl:
        :param swing_hl_v2:
        :param structure:
        :param show:
        :return:
        """
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

        if order_blocks:
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

        if swing_hl_v2:
            hl = self.swing_hl
            h = hl[hl['HighLow']==1]
            l = hl[hl['HighLow']==-1]

            fig.add_trace(go.Scatter(
                x=self.data['Date'].iloc[h.index],
                y=h['Level'],
                mode='markers',
                marker_symbol="triangle-up-dot",
                marker_size=10,
                name='Swing High',
                marker_color='green',
            ))
            fig.add_trace(go.Scatter(
                x=self.data['Date'].iloc[l.index],
                y=l['Level'],
                mode='markers',
                marker_symbol="triangle-down-dot",
                marker_size=10,
                name='Swing Low',
                marker_color='red',
            ))

        if structure:
            struct = self.structure_map
            struct.dropna(subset=['Level'], inplace=True)
            # for i in range(len(struct)):
            #     fig.add_shape(type="line",
            #         x0=self.data['Date'].iloc[struct.index[i]], x1=self.data['Date'].iloc[int(struct['BrokenIndex'].iloc[i])],
            #         y0=struct['Level'].iloc[i], y1=struct['Level'].iloc[i],
            #         label=dict(text="BOS" if np.isnan(struct['CHOCH'].iloc[i]) else "CHOCH"),
            #     )

            for i in range(len(struct)):
                x0 = self.data['Date'].iloc[struct.index[i]]
                x1 = self.data['Date'].iloc[int(struct['BrokenIndex'].iloc[i])]
                y = struct['Level'].iloc[i]
                label = "BOS" if np.isnan(struct['CHOCH'].iloc[i]) else "CHOCH"
                direction = struct[label].iloc[i]

                # Add scatter trace for the line
                fig.add_trace(go.Scatter(
                    x=[x0, x1],  # x-coordinates of the line
                    y=[y, y],  # y-coordinates of the line
                    mode="lines+text",  # Line and optional label
                    line=dict(color="blue" if label=="BOS" else "orange"),  # Customize line color
                    text=[None, label],  # Add label only at one end
                    textposition="top left" if direction==1 else "bottom left",  # Adjust label position
                    name=label,  # Legend entry for this line
                    showlegend=False
                ))
                # print(type(struct['CHOCH'].iloc[i]))

        fig.update_layout(xaxis_rangeslider_visible=False)
        if show:
            fig.show()
        return fig


def EMA(array, n):
    """
    :param array: price of the stock
    :param n: window size
    :type n: int
    :return: Exponential moving average
    :rtype: pd.Series
    """
    return pd.Series(array).ewm(span=n, adjust=False).mean()

def swing(data, length):
    pass

if __name__ == "__main__":
    from data_fetcher import fetch
    data = fetch('ICICIBANK.NS', period='1mo', interval='15m')
    data = fetch('RELIANCE.NS', period='1mo', interval='15m')
    data['Date'] = data.index.to_series()
    filter = pd.to_datetime('2024-12-17 09:50:00.0000000011',
               format='%Y-%m-%d %H:%M:%S.%f')
    # data = data[data['Date']<filter]
    # print(SMC(data).backtest_buy_signal())
    # print(SMC(data).swing_highs_lows_v3(10).to_string())
    # print(data.tail())
    SMC(data).plot(order_blocks=False, swing_hl=False, swing_hl_v2=True, structure=True, show=True)
    # struct = SMC(data).structure_map
    # print(struct)
    #
    # for i in range(len(data)):
    #     print(i, data['Date'][i], struct['BrokenIndex'].iloc[i])
    # SMC(data).structure()
