from dash import Dash, Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
from indicators import SMC
from data_fetcher import fetch
from strategies import SMC_test, plot_backtest

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

        html.Label("Swing High/Low Window Size", className="form-label"),
        dcc.Input(
            id="window",
            type="number",
            value=10,
            placeholder="Enter window size",
            className="form-control mb-3"
        ),

        dbc.Button("Run", id="submit-button", color="primary", className="w-100 mb-4"),
    ]),

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

# Callback for updating the visualizations
@callback(
    Input("submit-button", "n_clicks"),
    State("name", "value"),
    State("window", "value")
)
def update_visuals(n_clicks, name, window):
    if n_clicks <= 0 or not name:
        return

    # Clear existing files
    open('assets/backtest_results.html', 'w').close()
    open('assets/SMC.html', 'w').close()

    ticker = symbols[symbols['NAME OF COMPANY'] == name]['YahooEquiv'].values[0]
    data = fetch(ticker, '1mo', '15m')

    fig = SMC(data=data, swing_hl_window_sz=window).plot(show=False).update_layout(title=dict(text=ticker))

    plot_backtest(data, SMC_test, 'assets/backtest_results.html')
    fig.write_html('assets/SMC.html')

if __name__ == "__main__":
    # Clear initial files
    open('assets/backtest_results.html', 'w').close()
    open('assets/SMC.html', 'w').close()

    app.run(debug=True)