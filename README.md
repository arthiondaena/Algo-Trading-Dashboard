# Algorithmic Trading Dashboard

![License](https://img.shields.io/github/license/arthiondaena/backtesting.svg)
[![CI](https://github.com/arthiondaena/backtesting/actions/workflows/hf_push.yml/badge.svg)](https://github.com/.github/workflows/hf_push.yml)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Invicto69/Algo_Trading_Dashboard_streamlit)

## Description

This project is an **Algorithmic Trading Dashboard** designed to evaluate and backtest trading strategies on Nifty50
stocks. It provides a user-friendly interface powered by Streamlit, allowing users to select strategies, customize
parameters, and visualize backtest results. The dashboard supports multiple strategies, including **Order Block**, 
**Order Block with EMA**, and **Structure Trading**.

### Key features:

- Backtesting of trading strategies on Nifty50 stocks.
- Customizable parameters such as swing high/low window size, EMA lengths, and trade intervals.
- Visualization of backtest results, including equity curves and trade performance metrics.
- Support for multiprocessing to speed up backtesting across multiple stocks.

## Table of Contents

- [Huggingface Spaces Web demo](#Huggingface-Spaces-Web-demo)
- [Installation](#installation)
- [Usage](#usage)
- [Strategies](#strategies)
- [License](#license)

## Huggingface Spaces Web demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Streamlit](https://github.com/streamlit). Try it out [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/mae)


## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/arthiondaena/Algo-Trading-Dashboard.git
   cd Algo-Trading-Dashboard
   ```

2. Setup virtual environment (optional):
   ```bash
   python -m venv env 
   # Activation for linux
   source env/Scripts/activate
   # Activation for windows
   env/Scripts/activate.ps1
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Dashboard Features

1. **Complete Backtest**:
    - Run backtests on all Nifty50 stocks using a selected strategy.
    - Customize parameters such as swing high/low window size, EMA lengths, and trade intervals.
    - View aggregated results and detailed plots for individual stocks.

2. **Single Backtest**:
    - Run a backtest on a single stock with customizable parameters.
    - Visualize signal plots and backtest results.

### Running a Backtest

1. Select a strategy from the dropdown menu:
    - **Order Block**: Uses order blocks to generate buy/sell signals.
    - **Order Block with EMA**: Combines order blocks with EMA crossovers for signal generation.
    - **Structure Trading**: Focuses on market structure (e.g., breakouts and pullbacks).

2. Customize parameters such as:
    - Swing high/low window size.
    - EMA lengths (if applicable).
    - Trade interval and period.

3. Click the **Run** button to execute the backtest.

4. View the results, including:
    - Equity curves.
    - Trade performance metrics (e.g., return %, win rate, best/worst trade).
    - Signal plots for individual stocks.

## Strategies

### 1. Order Block

- Generates buy/sell signals based on order blocks.
- Sets stop-loss and take-profit levels at 5% of the entry price.

### 2. Order Block with EMA

- Combines order blocks with EMA crossovers for signal generation.
- Allows closing trades on EMA crossovers.

### 3. Structure Trading

- Focuses on market structure (e.g., breakouts and pullbacks).
- Uses trailing stop-loss to manage trades.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.