import streamlit as st
import pandas as pd
from streamlit.components import v1 as components
from utils import complete_test

def complete_backtest():
    st.title("Evaluate Strategy")

    limits = pd.read_csv('data/yahoo_limits.csv')
    period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    c1, c2 = st.columns(2)
    with c1:
        # Select strategy
        strategy = st.selectbox("Select Strategy", ['Order Block', 'Order Block with EMA', 'Structure trading'], index=2)
    with c2:
        # Swing High/Low window size
        swing_hl = st.number_input("Swing High/Low Window Size", min_value=1, value=10)

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
        st.session_state.results = complete_test(strategy, period, interval, swing_hl=swing_hl, ema1=ema1, ema2=ema2, cross_close=cross_close)

    if "results" in st.session_state:
        cols = ['stock', 'Start', 'End', 'Return [%]', 'Equity Final [$]', 'Buy & Hold Return [%]', '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]']
        df = st.dataframe(st.session_state.results, hide_index=True, column_order=cols, on_select="rerun", selection_mode="single-row")
        if df.selection.rows:
            row = df.selection.rows
            plot = st.session_state.results['plot'].values[row]
            components.html(plot[0], height=1067)

complete_backtest()