import gradio as gr
from indicators import SMC
from utils import smc_plot_backtest, smc_ema_plot_backtest, smc_structure_plot_backtest, fetch
import pandas as pd

symbols = pd.read_csv('data/Ticker_List_NSE_India.csv')
limits = pd.read_csv('data/yahoo_limits.csv')

def run(stock, interval, period, strategy, swing_hl, ema1=9, ema2=21, cross_close=False):
    # Downloading ticker data.
    ticker =  symbols[symbols['NAME OF COMPANY'] == stock]['YahooEquiv'].values[0]
    data = fetch(ticker, period, interval)

    # Plotting signal plot based on strategy.
    if strategy == "Order Block" or strategy == "Order Block with EMA":
        signal_plot = (SMC(data=data, swing_hl_window_sz=swing_hl).
                       plot(order_blocks=True, swing_hl=True, show=False).
                       update_layout(title=dict(text=ticker)))
    else:
        signal_plot = (SMC(data=data, swing_hl_window_sz=swing_hl).
                       plot(swing_hl_v2=True, structure=True, show=False).
                       update_layout(title=dict(text=ticker)))

    backtest_plot = gr.Plot()


    # Plotting backtest plot based on strategy.
    if strategy == "Order Block":
        backtest_plot = smc_plot_backtest(data, 'test.html', swing_hl)
    if strategy == "Order Block with EMA":
        backtest_plot = smc_ema_plot_backtest(data, 'test.html', ema1, ema2, cross_close)
    if strategy == "Structure trading":
        backtest_plot = smc_structure_plot_backtest(data, 'test.html', swing_hl)

    return signal_plot, backtest_plot


with gr.Blocks(fill_width=True) as app:
    gr.Markdown(
        '# Algorithmic Trading Dashboard'
    )
    stock = gr.Dropdown(symbols['NAME OF COMPANY'].unique().tolist(), label='Select Company', value=None)

    with gr.Row():
        interval = gr.Dropdown(limits['interval'].tolist(), label='Select Interval', value=None)

        period_list = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        period = gr.Dropdown(label = 'Select Period', choices=[])

        # Updating period based on interval
        def update_period(interval):
            limit = limits[limits['interval'] == interval]['limit'].values[0]
            idx = period_list.index(limit)
            return gr.Dropdown(period_list[:idx+1]+['max'], interactive=True, label='Select Period')

        interval.change(update_period, [interval], [period])

    with gr.Row():
        strategy = gr.Dropdown(['Order Block', 'Order Block with EMA', 'Structure trading'], label='Strategy', value=None)
        swing_hl = gr.Number(label="Swing High/Low Window Size", value=10, interactive=True)

    @gr.render(inputs=[strategy])
    def show_extra(strat):
        if strat == "Order Block with EMA":
            with gr.Row():
                ema1 = gr.Number(label='Fast EMA length', value=9)
                ema2 = gr.Number(label='Slow EMA length', value=21)
                cross_close = gr.Checkbox(label='Close trade on EMA crossover')
            input = [stock, interval, period, strategy, swing_hl, ema1, ema2, cross_close]

        elif strat == "Order Block" or strat == "Structure trading":
            input = [stock, interval, period, strategy, swing_hl]
        else:
            input = []

        btn.click(
            run,
            inputs=input,
            outputs=[signal_plot, backtest_plot]
        )

        examples = gr.Examples(
            examples=[
                ["Reliance Industries Limited", "15m", "max", "Order Block", 10],
                ["Reliance Industries Limited", "15m", "max", "Order Block with EMA", 10],
                ["Reliance Industries Limited", "15m", "max", "Structure trading", 20],
            ],
            example_labels=['Order Block', 'Order Block with EMA', 'Structure trading'],
            inputs=[stock, interval, period, strategy, swing_hl]
        )

    btn = gr.Button("Run")

    with gr.Row():
        signal_plot = gr.Plot(label='Signal plot')

    with gr.Row():
        backtest_plot = gr.Plot(label='Backtesting plot')

app.launch(debug=True)