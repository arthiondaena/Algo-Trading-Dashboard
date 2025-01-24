import pandas as pd
import streamlit as st
from streamlit.components import v1 as components

from indicators import SMC
from utils import fetch, run_strategy

def algorithmic_trading_dashboard():
    @st.cache_data
    def load_data():
        # Load data
        symbols = pd.read_csv('data/Ticker_List_NSE_India.csv')
        limits = pd.read_csv('data/yahoo_limits.csv')
        return symbols, limits

    symbols, limits = load_data()

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
        swing_hl = st.number_input("Swing High/Low Window Size", min_value=1, value=10,
                                   help = "Minimum window size for finding swing highs and lows.")

    # EMA parameters if "Order Block with EMA" is selected
    if strategy == "Order Block with EMA":
        c1, c2, c3 = st.columns(3)
        with c1:
            ema1 = st.number_input("Fast EMA Length", min_value=1, value=9,
                                   help = "Length of Fast moving Exponential Moving Average.")
        with c2:
            ema2 = st.number_input("Slow EMA Length", min_value=1, value=21,
                                   help = "Length of Slow moving Exponential Moving Average.")
        with c3:
            cross_close = st.checkbox("Close trade on EMA crossover", value=False)
    else:
        ema1, ema2, cross_close = None, None, None

    with st.expander("Advanced options"):
        c1, c2 = st.columns(2)
        with c1:
            initial_cash = st.number_input("Initial Cash [₹]", min_value=10000, value=10000)
        with c2:
            commission = st.number_input("Commission [%]", value = 0, min_value=-10, max_value=10,
                                         help="Commission is the commission ratio. E.g. if your broker's "
                                              "commission is 1% of trade value, set commission to 1.")

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

        backtest_results = run_strategy(ticker, strategy, period, interval,
                                swing_hl=swing_hl, ema1=ema1, ema2=ema2, cross_close=cross_close,
                                cash=initial_cash, commission=commission/100)

        color = "green" if backtest_results['Return [%]'].values[0] > 0 else "red"

        # Display plots
        st.write(f"### :{color}[Signal Plot]")
        st.plotly_chart(signal_plot, width=1200)

        st.write(f'### :{color}[Backtest Results]')
        cols = ['stock', 'Start', 'End', 'Return [%]', 'Equity Final [₹]', 'Buy & Hold Return [%]', '# Trades',
                'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]']
        st.dataframe(backtest_results, hide_index=True, column_order=cols)

        st.write(f"### :{color}[Backtest Plot]")
        plot = backtest_results['plot']
        components.html(plot[0], height=1067)

algorithmic_trading_dashboard()