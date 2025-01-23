import streamlit as st

st.set_page_config(page_title="Algorithmic Trading Dashboard", layout="wide", initial_sidebar_state="auto",
                       menu_items=None, page_icon=":chart_with_upwards_trend:")

dashboard = st.Page("pages/dashboard.py", title="Dashboard")
complete_test = st.Page("pages/complete_backtest.py", title="Nifty50 Test")

pg = st.navigation([dashboard, complete_test])

pg.run()