from dash import Dash, Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
from indicators import SMC
from data_fetcher import fetch
from strategies import smc_plot_backtest, smc_ema_plot_backtest

# Load symbols data
symbols = pd.read_csv('data/Ticker_List_NSE_India.csv')

# Initialize the app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to create the layout
app.layout = dbc.Container([
    dbc.Row([
        html.H1("Algorithmic Trading Dashboard", className="text-center mb-4")
    ]),

    dbc.Row([
        html.Label("Select Company Name", className="form-label"),
        dcc.Dropdown(
            id="name",
            options=[{"label": name, "value": name} for name in symbols['NAME OF COMPANY'].unique()],
            value='',
            placeholder="Select a company",
            className="mb-3"
        ),

        html.Label("Select Strategy", className="form-label"),
        dcc.Dropdown(
            id="strategy",
            options=['SMC', 'SMC with EMA'],
            value='',
            placeholder="Select Strategy",
            className="mb-3"
        ),

        html.Label("Swing High/Low Window Size", className="form-label"),
        dcc.Input(
            id="window",
            type="number",
            value=10,
            placeholder="Enter window size",
            className="form-control mb-3"
        ),

    ]),

    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Fast EMA Length: ", className="form-label"),
                dcc.Input(
                    id="ema1",
                    type="number",
                    value=9,
                    placeholder="Enter EMA Length",
                    # className="form-control mb-3"
                    className = "text-nowrap"
                ),
            ], md=8),

            dbc.Col([
                html.Label("Slow EMA Length: ", className="form-label"),
                dcc.Input(
                    id="ema2",
                    type="number",
                    value=21,
                    placeholder="Enter EMA size",
                    # className="form-control mb-3"
                    className="text-nowrap"
                ),

            dbc.Col([dcc.Checklist(['Close on EMA crossover'], id='closecross', className="text-nowrap")]),

            ]),
        ]),
        ], style={'display': 'block'}, id='smc_ema'
    ),

    dbc.Button("Run", id="submit-button", color="primary", className="w-100 mb-4"),

    dbc.Row([
        html.H5("Order Block Chart", className="text-center mb-3"),
        html.Iframe(
            src="assets/SMC.html",
            style={"height": "450px", "width": "95%", "border": "none"},
            className="mb-4"
        ),

        html.H5("Backtest Results", className="text-center mb-3"),
        html.Iframe(
            src="assets/backtest_results.html",
            style={"height": "1067px", "width": "95%", "border": "none"}
        ),
    ])
], fluid=True)


@callback(
    Output("smc_ema", 'style'),
    Input("strategy", 'value')
)
def update_layout(strategy):
    if strategy=='SMC with EMA':
        return {'display':'block'}
    else:
        return {'display':'none'}


# Callback for updating the visualizations
@callback(
    Input("submit-button", "n_clicks"),
    State("name", "value"),
    State("window", "value"),
    State("strategy", "value"),
    State("ema1", "value"),
    State("ema2", "value"),
    State("closecross", "value")
)
def update_visuals(n_clicks, name, window, strategy, ema1, ema2, closecross):
    if n_clicks <= 0 or not name:
        return

    # Clear existing files
    open('assets/backtest_results.html', 'w').close()
    open('assets/SMC.html', 'w').close()

    ticker = symbols[symbols['NAME OF COMPANY'] == name]['YahooEquiv'].values[0]
    data = fetch(ticker, '1mo', '15m')

    fig = SMC(data=data, swing_hl_window_sz=window).plot(show=False).update_layout(title=dict(text=ticker))

    print(strategy)
    if strategy=='SMC':
        smc_plot_backtest(data, 'assets/backtest_results.html', swing_hl=window)
    elif strategy=='SMC with EMA':
        smc_ema_plot_backtest(data, 'assets/backtest_results.html', ema1, ema2, closecross)

    fig.write_html('assets/SMC.html')

if __name__ == "__main__":
    # Clear initial files
    open('assets/backtest_results.html', 'w').close()
    open('assets/SMC.html', 'w').close()

    app.run(debug=True)