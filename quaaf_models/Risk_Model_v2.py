import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import statistics
from scipy.stats import skew, kurtosis
from dateutil.relativedelta import relativedelta

# -----------------------------
# Data download and processing
# -----------------------------
def get_adj_close_prices(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if 'Close' in data.columns and not data.empty:
            df = data[['Close']].reset_index()
            return df
        else:
            raise ValueError("Close prices not found or data is empty.")
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def calculate_ln_returns(df):
    """
    Calculate log returns in percentage and assign directly to df['Target Return (%)'].
    The first row is set to None if there's at least one row.
    """
    if df.empty:
        # If there's no data, just return the empty df
        return df

    # Calculate log returns
    df['Target Return (%)'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    
    # If there's at least 1 row, set the first row's return to None
    if len(df) >= 1:
        df.loc[df.index[0], 'Target Return (%)'] = None
    
    return df

def add_target_drawdown(data):
    try:
        data['Target Drawdown'] = data['Close'] - data['Close'].expanding().max()
    except Exception as e:
        print(f"Error calculating drawdown: {e}")
    return data

def add_consecutive_drawdown_count(data):
    drawdown_count = 0
    counts = []
    for dd in data['Target Drawdown']:
        if dd != 0:
            drawdown_count += 1
        else:
            drawdown_count = 0
        counts.append(drawdown_count)
    data['Consecutive Drawdown Count'] = counts
    return data

def add_target_drawdown_ratio(data):
    try:
        data['Cumulative Max'] = data['Close'].cummax()
        data['Target Drawdown Ratio (%)'] = data['Target Drawdown'] / data['Cumulative Max'] * 100
        data.drop(columns=['Cumulative Max'], inplace=True)
    except Exception as e:
        print(f"Error calculating drawdown ratio: {e}")
    return data

def add_target_price_ratio(data):
    try:
        initial_price = data['Close'].iloc[0]
        data['Target Price Ratio (%)'] = data['Close'] / initial_price * 100
    except Exception as e:
        print(f"Error calculating target price ratio: {e}")
    return data

# -----------------------------
# Metrics functions (per period)
# -----------------------------
def calculate_geomean(data, tenor):
    try:
        if 'Target Geodummy (%)' not in data.columns:
            raise ValueError("Missing 'Target Geodummy (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Geodummy (%)'].iloc[-tenor:].reset_index(drop=True)
        return statistics.geometric_mean(obs) - 100
    except Exception as e:
        print(f"Error in geomean: {e}")
        return None

def calculate_median(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("Missing 'Target Return (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        return statistics.median(obs)
    except Exception as e:
        print(f"Error in median: {e}")
        return None

def calculate_cum_return(data, tenor):
    try:
        if 'Target Price' not in data.columns:
            raise ValueError("Missing 'Target Price'")
        if tenor <= 0 or tenor > len(data):
            tenor = len(data)
        obs = data['Target Price'].iloc[-tenor:].reset_index(drop=True)
        first = obs.iloc[0]
        last = obs.iloc[-1]
        if first == 0:
            return "N/A"
        return (last - first) / first
    except Exception as e:
        print(f"Error in cumulative return: {e}")
        return None

def calculate_std(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("Missing 'Target Return (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        return obs.std(ddof=1)
    except Exception as e:
        print(f"Error in std: {e}")
        return None

def calculate_skew(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("Missing 'Target Return (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        return obs.skew()
    except Exception as e:
        print(f"Error in skew: {e}")
        return None

def calculate_kurtosis(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("Missing 'Target Return (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        return obs.kurtosis()
    except Exception as e:
        print(f"Error in kurtosis: {e}")
        return None

def calculate_min_target_return(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("Missing 'Target Return (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        return obs.min()
    except Exception as e:
        print(f"Error in min return: {e}")
        return None

def calculate_max_target_return(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("Missing 'Target Return (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        return obs.max()
    except Exception as e:
        print(f"Error in max return: {e}")
        return None

def calculate_avg_drawdown(data, tenor):
    try:
        if 'Target Drawdown ($)' not in data.columns:
            raise ValueError("Missing 'Target Drawdown ($)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Drawdown ($)'].iloc[-tenor:].reset_index(drop=True)
        return statistics.mean(obs)
    except Exception as e:
        print(f"Error in avg drawdown: {e}")
        return None

def calculate_avg_drawdown_percentage(data, tenor):
    try:
        if 'Target Drawdown Ratio (%)' not in data.columns:
            raise ValueError("Missing 'Target Drawdown Ratio (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Drawdown Ratio (%)'].iloc[-tenor:].reset_index(drop=True)
        return statistics.mean(obs)
    except Exception as e:
        print(f"Error in avg drawdown percentage: {e}")
        return None

def calculate_drawdown_length(data, tenor):
    try:
        if 'Consecutive Drawdown Count' not in data.columns:
            raise ValueError("Missing 'Consecutive Drawdown Count'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Consecutive Drawdown Count'].iloc[-tenor:].reset_index(drop=True)
        return obs.max()
    except Exception as e:
        print(f"Error in drawdown length: {e}")
        return None

def calculate_max_drawdown(data, tenor):
    try:
        if 'Target Drawdown Ratio (%)' not in data.columns:
            raise ValueError("Missing 'Target Drawdown Ratio (%)'")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        obs = data['Target Drawdown Ratio (%)'].iloc[-tenor:].reset_index(drop=True)
        return obs.min()
    except Exception as e:
        print(f"Error in max drawdown: {e}")
        return None

def calculate_benchmark_corr(df1, df2, tenor):
    try:
        if 'Target Price' not in df1.columns or 'Target Price' not in df2.columns:
            raise KeyError("Missing 'Target Price' in one of the dataframes.")
        if tenor <= 0 or tenor > len(df1) or tenor > len(df2):
            tenor = min(len(df1), len(df2))
        series1 = df1['Target Price'].iloc[-tenor:].ffill().reset_index(drop=True)
        series2 = df2['Target Price'].iloc[-tenor:].ffill().reset_index(drop=True)
        return series1.corr(series2)
    except Exception as e:
        print(f"Error in benchmark correlation: {e}")
        return None

def information_ratio(df1, df2, tenor):
    try:
        if 'Target Return (%)' not in df1.columns or 'Target Return (%)' not in df2.columns:
            raise KeyError("Missing 'Target Return (%)' in one of the dataframes.")
        if tenor <= 0 or tenor > len(df1) or tenor > len(df2):
            tenor = min(len(df1), len(df2))
        ret1 = df1['Target Return (%)'].iloc[-tenor:].ffill().reset_index(drop=True)
        ret2 = df2['Target Return (%)'].iloc[-tenor:].ffill().reset_index(drop=True)
        active_return = ret1 - ret2
        tracking_error = np.std(active_return, ddof=1)
        if tracking_error == 0:
            return np.nan
        return np.mean(active_return) / tracking_error
    except Exception as e:
        print(f"Error in Information Ratio: {e}")
        return np.nan

# -----------------------------
# Day count helper functions
# -----------------------------
def get_business_day_months_ago_in_data(data, months):
    end_date = data['Target Date'].max()
    start_date = end_date - relativedelta(months=months)
    if start_date < data['Target Date'].min():
        start_date = data['Target Date'].min()
    start_date = pd.Timestamp(start_date)
    if start_date.weekday() >= 5:
        start_date = pd.offsets.BDay().rollback(start_date)
    while start_date not in data['Target Date'].values:
        start_date -= pd.Timedelta(days=1)
    return len(data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)])

def get_business_day_years_ago_in_data(data, years):
    end_date = data['Target Date'].max()
    start_date = end_date - relativedelta(years=years)
    if start_date < data['Target Date'].min():
        start_date = data['Target Date'].min()
    start_date = pd.Timestamp(start_date)
    if start_date.weekday() >= 5:
        start_date = pd.offsets.BDay().rollback(start_date)
    while start_date not in data['Target Date'].values:
        start_date -= pd.Timedelta(days=1)
    return len(data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)])

def get_business_day_start_of_year_in_data(data):
    current_year = datetime.now().year
    start_date = pd.Timestamp(f'{current_year}-01-01')
    end_date = data['Target Date'].max()
    if start_date.weekday() >= 5:
        start_date = pd.offsets.BDay().rollback(start_date)
    while start_date not in data['Target Date'].values:
        start_date += pd.Timedelta(days=1)
    return len(data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)])

def calculate_inception_days(data):
    first_date = data['Target Date'].min()
    last_date = data['Target Date'].max()
    return len(data[(data['Target Date'] >= first_date) & (data['Target Date'] <= last_date)])

# -----------------------------
# Main processing per ticker
# -----------------------------
def process_single_ticker(ticker, benchmark1_df, benchmark2_df):
    print(f"\nProcessing {ticker} ...")
    prices = get_adj_close_prices(ticker)
    if prices is None or prices.empty:
        print(f"No data for {ticker}. Skipping...")
        return None, None

    # Calculate log returns and update prices in place
    prices = calculate_ln_returns(prices)

    # Rename columns and create target_df
    target_df = prices.rename(columns={'Date': 'Target Date', 'Close': 'Target Price'}).copy()
    target_df['Target Geodummy (%)'] = target_df['Target Return (%)'].apply(lambda x: (x + 100) if x is not None else None)

    # Calculate drawdowns and related measures
    target_df = add_target_drawdown(target_df)
    target_df['Target Drawdown ($)'] = target_df['Target Drawdown']
    target_df = add_consecutive_drawdown_count(target_df)
    target_df = add_target_drawdown_ratio(target_df)
    target_df['Target Drawdown Ratio (%)'] = target_df['Target Drawdown Ratio (%)']
    target_df = add_target_price_ratio(target_df)
    target_df['Target Price Ratio (%)'] = target_df['Target Price Ratio (%)']

    # Define periods and compute day counts for target and benchmarks
    period_funcs = {
        '1 Month': lambda df: get_business_day_months_ago_in_data(df, 1),
        '3 Month': lambda df: get_business_day_months_ago_in_data(df, 3),
        '6 Month': lambda df: get_business_day_months_ago_in_data(df, 6),
        'YTD': lambda df: get_business_day_start_of_year_in_data(df),
        '1 Year': lambda df: get_business_day_years_ago_in_data(df, 1),
        '2 Year': lambda df: get_business_day_years_ago_in_data(df, 2),
        '3 Year': lambda df: get_business_day_years_ago_in_data(df, 3),
        '5 Year': lambda df: get_business_day_years_ago_in_data(df, 5),
        'Since Inception': lambda df: calculate_inception_days(df)
    }

    day_counts = {p: func(target_df) for p, func in period_funcs.items()}
    day_counts_b1 = {p: func(benchmark1_df) for p, func in period_funcs.items()}
    day_counts_b2 = {p: func(benchmark2_df) for p, func in period_funcs.items()}

    # Compute metrics for each period
    metrics = {}
    for period, days in day_counts.items():
        if days == "N/A" or days is None or days <= 1:
            continue
        tenor_adj = days
        if period in ['5 Year', 'Since Inception']:
            tenor_adj = days - 1

        metrics.setdefault("Correlation with Benchmark #1", {})[period] = calculate_benchmark_corr(target_df, benchmark1_df, day_counts_b1[period])
        metrics.setdefault("Correlation with Benchmark #2", {})[period] = calculate_benchmark_corr(target_df, benchmark2_df, day_counts_b2[period])
        metrics.setdefault("Geometric Mean Return (Annualized)", {})[period] = calculate_geomean(target_df, tenor_adj)
        metrics.setdefault("Median Fund Return", {})[period] = calculate_median(target_df, days)
        metrics.setdefault("Cumulative Return", {})[period] = calculate_cum_return(target_df, tenor_adj)
        if period in ['1 Month', '3 Month', '6 Month', 'YTD']:
            std_adj = (days ** 0.5) / 100
        else:
            std_adj = (252 ** 0.5) / 100
        raw_std = calculate_std(target_df, days)
        metrics.setdefault("Standard Deviation (Annualized)", {})[period] = raw_std * std_adj if raw_std not in [None, "N/A"] else raw_std
        metrics.setdefault("Kurtosis", {})[period] = calculate_kurtosis(target_df, days)
        metrics.setdefault("Skewness", {})[period] = calculate_skew(target_df, days)
        metrics.setdefault("Minimum Return", {})[period] = calculate_min_target_return(target_df, days)
        metrics.setdefault("Maximum Return", {})[period] = calculate_max_target_return(target_df, days)
        metrics.setdefault("Average Drawdown Per Share", {})[period] = calculate_avg_drawdown(target_df, days)
        metrics.setdefault("Average Drawdown Percentage", {})[period] = calculate_avg_drawdown_percentage(target_df, days)
        metrics.setdefault("Information Ratio", {})[period] = information_ratio(target_df, benchmark1_df, day_counts_b1[period])
        metrics.setdefault("Length of Drawdown", {})[period] = calculate_drawdown_length(target_df, days)
        metrics.setdefault("Maximum Drawdown", {})[period] = calculate_max_drawdown(target_df, days)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.index.name = "Metric"

    return target_df, metrics_df

# -----------------------------
# Main function
# -----------------------------
def main():
    print("Welcome to the Risk Model Script.")
    tickers_input = input("Enter a list of tickers separated by commas (e.g., AAPL, MSFT, TSLA): ")
    benchmarks_input = input("Enter two benchmark tickers separated by commas (e.g., ^GSPC, ^DJI): ")

    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    benchmarks = [b.strip().upper() for b in benchmarks_input.split(',') if b.strip()]

    if len(benchmarks) < 2:
        print("Please enter exactly two benchmark tickers.")
        return
    if len(tickers) < 1:
        print("Please enter at least one ticker.")
        return

    benchmark1_df_raw = get_adj_close_prices(benchmarks[0])
    benchmark2_df_raw = get_adj_close_prices(benchmarks[1])
    if benchmark1_df_raw is None or benchmark2_df_raw is None:
        print("Error retrieving benchmark data. Exiting.")
        return

    benchmark1_df = benchmark1_df_raw.rename(columns={'Date': 'Target Date', 'Close': 'Target Price'}).copy()
    benchmark1_df = calculate_ln_returns(benchmark1_df)
    benchmark2_df = benchmark2_df_raw.rename(columns={'Date': 'Target Date', 'Close': 'Target Price'}).copy()
    benchmark2_df = calculate_ln_returns(benchmark2_df)

    sheets = {}
    for ticker in tickers:
        raw_df, metrics_df = process_single_ticker(ticker, benchmark1_df, benchmark2_df)
        if raw_df is None:
            continue
        sheets[f"{ticker} - Raw Data"] = raw_df
        sheets[f"{ticker} - Metrics"] = metrics_df

    if not sheets:
        print("No data to output.")
        return

    output_filename = "risk_model_results.xlsx"
    try:
        with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        print(f"\nAll results have been saved to '{output_filename}'.")
    except Exception as e:
        print(f"Error writing Excel file: {e}")

if __name__ == "__main__":
    main()
