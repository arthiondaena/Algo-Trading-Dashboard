import streamlit as st
from src.colorer import get_logger

def app():
    st.set_page_config(page_title="Algorithmic Trading Dashboard", layout="wide", initial_sidebar_state="auto",
                           menu_items=None, page_icon=":chart_with_upwards_trend:")

    if "logger" not in st.session_state:
        st.session_state.logger = get_logger()

    single_test = st.Page("page/single_backtest.py", title="Run Strategy")
    complete_test = st.Page("page/complete_backtest.py", title="Evaluate Strategy")

    pg = st.navigation({'Algorithmic Trading Dashboard':[single_test, complete_test]})

    pg.run()

if __name__ == "__main__":
    # run_logging()
    app()