import streamlit as st
import pandas as pd
import time

from streamlit.components import v1 as components
from src.utils import complete_test

def complete_backtest():
    @st.cache_data
    def load_data():
        # Load data
        limits = pd.read_csv('data/yahoo_limits.csv')
        return limits

    limits = load_data()

    st.markdown(
        """
        # Algorithmic Trading Dashboard
        ## Evaluate Strategy
        """
    )

    st.info("Strategy runs on most of the Nifty50 stocks",  icon="ℹ️")

    period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    c1, c2 = st.columns(2)
    with c1:
        # Select strategy
        strategy = st.selectbox("Select Strategy", ['Order Block', 'Order Block with EMA', 'Structure trading'], index=2)
    with c2:
        # Swing High/Low window size
        swing_hl = st.number_input("Swing High/Low Window Size", min_value=1, value=10,
                                   help = "Minimum window size for finding swing highs and lows.")

    c1, c2 = st.columns(2)
    with c1:
        # Select interval
        interval = st.selectbox("Select Interval", limits['interval'].tolist(), index=3)
    with c2:
        # Update period options based on interval
        limit = limits[limits['interval'] == interval]['limit'].values[0]
        idx = period_list.index(limit)
        period_options = period_list[:idx + 1] + ['max']
        period = st.selectbox("Select Period", period_options, index=2)

    # EMA parameters if "Order Block with EMA" is selected
    if strategy == "Order Block with EMA":
        c1, c2, c3 = st.columns([2, 2, 1.5])
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
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            initial_cash = st.number_input("Initial Cash [₹]", min_value=10000, value=10000)
        with c2:
            commission = st.number_input("Commission [%]", value = 0, min_value=-10, max_value=10,
                                         help="Commission is the commission ratio. E.g. if your broker's "
                                              "commission is 1% of trade value, set commission to 1.")
        with c3:
            multiprocess = st.checkbox("Multiprocess", value=True,
                                       help="Use multiple CPUs (if available) to parallelize the run. "
                                            "Run time is inversely proportional to no of CPUs available.")

    # Button to run the analysis
    if st.button("Run"):
        start = time.time()
        st.session_state.results = complete_test(strategy, period, interval, multiprocess,
                                        swing_hl=swing_hl, ema1=ema1, ema2=ema2,
                                        cross_close=cross_close, cash=initial_cash,
                                        commission=commission/100)
        st.success(f"Analysis finished in {round(time.time()-start, 2)} seconds")

    if "results" in st.session_state:
        # st.write("⬇️ Select a row in index column to get detailed information of the respective stock run.")
        st.markdown(f"""
                    ### :orange[Nifty50 stocks backtest result by using {strategy}]
                     ⬇️ Select a row in index column to get detailed information of the respective stock run.
                    """)
        cols = ['stock', 'Start', 'End', 'Return [%]', 'Equity Final [₹]', 'Buy & Hold Return [%]', '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]']
        df = st.dataframe(st.session_state.results, hide_index=True, column_order=cols, on_select="rerun", selection_mode="single-row")
        df.selection.rows = 1
        if df.selection.rows:
            row = df.selection.rows
            ticker = st.session_state.results['stock'].values[row]
            plot = st.session_state.results['plot'].values[row]
            color = "green" if st.session_state.results['Return [%]'].values[row][0] > 0 else "red"
            st.markdown(f"""
            ### :{color}[{ticker[0]} backtest plot] 📊
            """)
            components.html(plot[0], height=1067)

complete_backtest()