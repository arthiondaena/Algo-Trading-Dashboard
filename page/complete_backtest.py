import streamlit as st
import pandas as pd
import numpy as np
import time

from streamlit.components import v1 as components
from src.utils import complete_test, categorize_df

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

    stock_list = st.selectbox("Select Stock list", ['Nifty 50', 'Nifty Next 50', 'Nifty 100', 'Nifty 200'], index=0)

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
            close_on_crossover = st.checkbox("Close trade on EMA crossover", value=False)
    else:
        ema1, ema2, close_on_crossover = None, None, None

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
        st.session_state.results = complete_test(stock_list, strategy, period, interval, multiprocess,
                                        swing_hl=swing_hl, ema1=ema1, ema2=ema2,
                                        close_on_crossover=close_on_crossover, cash=initial_cash,
                                        commission=commission/100)
        st.success(f"Analysis finished in {round(time.time()-start, 2)} seconds")

    if "results" in st.session_state:
        # st.write("⬇️ Select a row in index column to get detailed information of the respective stock run.")
        st.markdown(f"""
                    ---
                    ### :orange[{stock_list} stocks backtest result by using {strategy} strategy]
                     ⬇️ Select rows in 'Select' column to get backtest plots of the selected stocks.
                    """)
        st.session_state.results['Select'] = False

        cols = ['Select', 'Stock', 'Sector', 'Start', 'End', 'Return [%]', 'Equity Final [₹]', 'Buy & Hold Return [%]', '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]']
        st.session_state.categorized_results = categorize_df(st.session_state.results, 'Sector', 'Return [%]')

        st.session_state.categorized_results_dict = {}
        st.session_state.selected_stocks = {}

        for category, df in st.session_state.categorized_results.items():
            mean = round(df['Return [%]'].mean(), 2)
            color = "green" if mean > 0 else "red"
            with st.expander(f"{str(category).upper()}    :{color}[Average return rate: {mean} %]"):
                st.session_state.categorized_results_dict[category] = (
                              st.data_editor(df,
                                  column_config={
                                      'Select': st.column_config.CheckboxColumn(
                                          'Select',
                                          default=False
                                      )
                                  },
                                  hide_index=True, column_order=cols,
                                  # on_select="rerun", selection_mode="single-row"
                              ))
                st.session_state.selected_stocks[category] = (
                    np.where(st.session_state.categorized_results_dict[category]['Select']))[0]

        for selected_rows in st.session_state.selected_stocks.values():
            if len(selected_rows) > 0:
                st.toast("Scroll to the bottom of page to view backtest plots.", icon=":material/vertical_align_bottom:")
                st.markdown(f"""
                            ---
                            ### :orange[Selected stocks backtest plots by using {strategy} strategy]
                            """)
                break

        for selected_rows in st.session_state.selected_stocks.values():
            for row in selected_rows:
                ticker = st.session_state.results['Stock'].values[row]
                plot = st.session_state.results['plot'].values[row]
                color = "green" if st.session_state.results['Return [%]'].values[row] > 0 else "red"
                with st.expander(f":{color}[{ticker} backtest plot] 📊"):
                    components.html(plot, height=900)

complete_backtest()