import pandas as pd
import streamlit as st
import os
import random
from bokeh.io import output_file, save
from bokeh.plotting import figure
from streamlit.components import v1 as components

from indicators import SMC
from utils import fetch, run_strategy

def use_file_for_bokeh(chart: figure, chart_height=1067):
    # Function used to replace st.boken_chart, because streamlit doesn't support bokeh v3
    file_name = f'bokeh_graph_{random.getrandbits(8)}.html'
    output_file(file_name)
    save(chart)
    with open(file_name, 'r', encoding='utf-8') as f:
        html = f.read()
    os.remove(file_name)
    components.html(html, height=chart_height)

st.bokeh_chart = use_file_for_bokeh

def algorithmic_trading_dashboard():
    # Load data
    symbols = pd.read_csv('data/Ticker_List_NSE_India.csv')
    limits = pd.read_csv('data/yahoo_limits.csv')

    # Dropdown options
    period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    st.markdown(
        """
        # Algorithmic Trading Dashboard
        ## Run Strategy
        """
    )

    # Input fields on the main page

    # Select stock
    stock = st.selectbox("Select Company", symbols['NAME OF COMPANY'].unique(), index=None)

    c1, c2 = st.columns(2)

    with c1:
        # Select interval
        interval = st.selectbox("Select Interval", limits['interval'].tolist(), index=3)

    with c2:
        # Update period options based on interval
        limit = limits[limits['interval'] == interval]['limit'].values[0]
        idx = period_list.index(limit)
        period_options = period_list[:idx + 1] + ['max']
        period = st.selectbox("Select Period", period_options, index=3)

    c1, c2 = st.columns(2)

    with c1:
        # Select strategy
        strategy = st.selectbox("Select Strategy", ['Order Block', 'Order Block with EMA', 'Structure trading'], index=2)

    with c2:
        # Swing High/Low window size
        swing_hl = st.number_input("Swing High/Low Window Size", min_value=1, value=10)

    # EMA parameters if "Order Block with EMA" is selected
    if strategy == "Order Block with EMA":
        c1, c2, c3 = st.columns(3)
        with c1:
            ema1 = st.number_input("Fast EMA Length", min_value=1, value=9)
        with c2:
            ema2 = st.number_input("Slow EMA Length", min_value=1, value=21)
        with c3:
            cross_close = st.checkbox("Close trade on EMA crossover", value=False)
    else:
        ema1, ema2, cross_close = None, None, None

    # Button to run the analysis
    if st.button("Run"):
        # Fetch ticker data
        ticker = symbols[symbols['NAME OF COMPANY'] == stock]['YahooEquiv'].values[0]
        data = fetch(ticker, period, interval)

        # Generate signal plot based on strategy
        if strategy == "Order Block" or strategy == "Order Block with EMA":
            signal_plot = (
                SMC(data=data, swing_hl_window_sz=swing_hl)
                .plot(order_blocks=True, swing_hl=True, show=False)
                .update_layout(title=dict(text=ticker))
            )
        else:
            signal_plot = (
                SMC(data=data, swing_hl_window_sz=swing_hl)
                .plot(swing_hl_v2=True, structure=True, show=False)
                .update_layout(title=dict(text=ticker))
            )

        # Generate backtest plot
        # if strategy == "Order Block":
        #     backtest_plot = smc_plot_backtest(data, 'test.html', swing_hl)
        # elif strategy == "Order Block with EMA":
        #     backtest_plot = smc_ema_plot_backtest(data, 'test.html', ema1, ema2, cross_close)
        # elif strategy == "Structure trading":
        #     backtest_plot = smc_structure_plot_backtest(data, 'test.html', swing_hl)

        backtest_results = run_strategy(ticker, strategy, period, interval, swing_hl=swing_hl, ema1=ema1, ema2=ema2, cross_close=cross_close)

        # Display plots
        st.write("### Signal Plot")
        st.plotly_chart(signal_plot, width=1200)

        st.write('### Backtest Results')
        cols = ['stock', 'Start', 'End', 'Return [%]', 'Equity Final [$]', 'Buy & Hold Return [%]', '# Trades',
                'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]']
        st.dataframe(backtest_results, hide_index=True, column_order=cols)

        st.write("### Backtest Plot")
        plot = backtest_results['plot']
        components.html(plot[0], height=1067)

algorithmic_trading_dashboard()