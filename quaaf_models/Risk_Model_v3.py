import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import statistics
from scipy.stats import skew, kurtosis
from dateutil.relativedelta import relativedelta

# Helper function to force a value to a scalar float.
def force_scalar(x):
    """
    Converts x to a single float if x is an array, list, or already numeric.
    Returns NaN if it can't convert.
    """
    # If it's already a numeric type, return it.
    if isinstance(x, (float, int)):
        return float(x)
    
    # If it's a list, take the first element.
    if isinstance(x, list):
        if len(x) > 0:
            return force_scalar(x[0])
        else:
            return np.nan

    # If it's a numpy array, flatten it.
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.item())
        else:
            return float(x.flatten()[0])
    
    # Otherwise, try to convert directly.
    try:
        return float(x)
    except Exception:
        return np.nan

def get_adj_close_prices(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'))
        if 'Close' in data.columns:
            adj_close_prices = data[['Close']].copy()
            adj_close_prices.reset_index(inplace=True)
            adj_close_prices.loc[:, 'Date'] = pd.to_datetime(adj_close_prices['Date'])
            return adj_close_prices
        else:
            raise ValueError("Adjusted Close prices not found in the data.")
    except Exception as e:
        print(f"An error occurred in get_adj_close_prices: {e}")
        return None

def calculate_ln_returns(data):
    ln_returns = []
    close = data['Close'].values
    for i in range(1, len(close)):
        ln_return = float((np.log(close[i] / close[i - 1]) * 100).item())
        ln_returns.append(ln_return)
    return ln_returns

def add_target_drawdown(data):
    try:
        if 'Close' not in data.columns:
            raise ValueError("The 'Target Price' column is missing from the DataFrame.")
        data['Target Drawdown'] = data['Close'] - data['Close'].expanding().max()
        return data
    except Exception as e:
        print(f"An error occurred while calculating target drawdown: {e}")
        return data

def add_consecutive_drawdown_count(data):
    try:
        drawdown_count = 0
        def count_consecutive(row):
            nonlocal drawdown_count
            if row != 0:
                drawdown_count += 1
            else:
                drawdown_count = 0
            return drawdown_count
        data['Consecutive Drawdown Count'] = data['Target Drawdown'].apply(count_consecutive)
        return data
    except Exception as e:
        print(f"An error occurred while calculating consecutive drawdown count: {e}")
        return data

def add_target_drawdown_ratio(data):
    try:
        data['Cumulative Max Target Price'] = data['Close'].cummax()
        data['Target Drawdown Ratio'] = data['Target Drawdown'] / data['Cumulative Max Target Price'] * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            data.drop(columns=['Cumulative Max Target Price'], inplace=True)
        return data
    except Exception as e:
        print(f"An error occurred while calculating Target Drawdown Ratio: {e}")
        return data

def add_target_price_ratio(data):
    try:
        initial_target_price = data['Close'].iloc[0]
        data['Target Price Ratio'] = data['Close'] / initial_target_price * 100
        return data
    except Exception as e:
        print(f"An error occurred while calculating Target Price Ratio: {e}")
        return data

def calculate_geomean(data, tenor):
    try:
        if 'Target Geodummy (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Geodummy (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Geodummy (%)'].iloc[-tenor:].reset_index(drop=True)
        obs = pd.to_numeric(observation_period, errors='coerce').dropna().tolist()
        if len(obs) == 0:
            return "N/A"
        return statistics.geometric_mean(obs) - 100
    except Exception as e:
        print(f"An error occurred in calculate_geomean: {e}")
        return None

def calculate_median(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        obs = pd.to_numeric(observation_period, errors='coerce').dropna().tolist()
        if len(obs) == 0:
            return "N/A"
        return statistics.median(obs)
    except Exception as e:
        print(f"An error occurred in calculate_median: {e}")
        return None

def calculate_cum_return(data, tenor):
    try:
        if 'Target Price' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Price' column.")
        if tenor <= 0 or tenor > len(data):
            tenor = len(data)
        series_prices = data['Target Price'].iloc[-tenor:].reset_index(drop=True)
        first_value = float(series_prices.iloc[0])
        last_value = float(series_prices.iloc[-1])
        if first_value == 0:
            return "N/A"
        percentage_change = (last_value - first_value) / first_value
        return percentage_change
    except Exception as e:
        print(f"An error occurred in calculate_cum_return: {e}")
        return None

def calculate_std(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return observation_period.std(ddof=1)
    except Exception as e:
        print(f"An error occurred while calculating standard deviation: {e}")
        return None

def calculate_skew(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return observation_period.skew()
    except Exception as e:
        print(f"An error occurred in calculate_skew: {e}")
        return None

def calculate_kurtosis(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return observation_period.kurtosis()
    except Exception as e:
        print(f"An error occurred in calculate_kurtosis: {e}")
        return None

def calculate_min_target_return(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return observation_period.min()
    except Exception as e:
        print(f"An error occurred in calculate_min_target_return: {e}")
        return None

def calculate_max_target_return(data, tenor):
    try:
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Return (%)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return observation_period.max()
    except Exception as e:
        print(f"An error occurred in calculate_max_target_return: {e}")
        return None

def calculate_avg_drawdown(data, tenor):
    try:
        if 'Target Drawdown ($)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown ($)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Drawdown ($)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return statistics.mean(observation_period.dropna())
    except Exception as e:
        print(f"An error occurred in calculate_avg_drawdown: {e}")
        return None

def calculate_avg_drawdown_percentage(data, tenor):
    try:
        if 'Target Drawdown Recovery Ratio (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown Recovery Ratio (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Drawdown Recovery Ratio (%)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return statistics.mean(observation_period.dropna())
    except Exception as e:
        print(f"An error occurred in calculate_avg_drawdown_percentage: {e}")
        return None

def calculate_drawdown_length(data, tenor):
    try:
        if 'Target Drawdown Recovery Length (Days)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown Recovery Length (Days)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Drawdown Recovery Length (Days)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return observation_period.max()
    except Exception as e:
        print(f"An error occurred in calculate_drawdown_length: {e}")
        return None

def calculate_max_drawdown(data, tenor):
    try:
        if 'Target Drawdown Recovery Ratio (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown Recovery Ratio (%)' column.")
        if tenor <= 0 or tenor > len(data):
            return "N/A"
        observation_period = data['Target Drawdown Recovery Ratio (%)'].iloc[-tenor:].reset_index(drop=True)
        observation_period = pd.to_numeric(observation_period, errors='coerce')
        return observation_period.min()
    except Exception as e:
        print(f"An error occurred in calculate_max_drawdown: {e}")
        return None

def calculate_benchmark_corr(df1, df2, tenor):
    try:
        if 'Target Price' not in df1.columns or 'Target Price' not in df2.columns:
            raise KeyError("Both DataFrames must contain a 'Target Price' column.")
        if tenor <= 0 or tenor > len(df1) or tenor > len(df2):
            tenor = min(len(df1), len(df2))
        series1 = df1['Target Price'].iloc[-tenor:].ffill().reset_index(drop=True)
        series2 = df2['Target Price'].iloc[-tenor:].ffill().reset_index(drop=True)
        series1 = pd.to_numeric(series1, errors='coerce')
        series2 = pd.to_numeric(series2, errors='coerce')
        if len(series1) != len(series2):
            raise ValueError("Mismatch in lengths of the two series for the specified tenor.")
        return series1.corr(series2)
    except Exception as e:
        print(f"An error occurred in calculate_benchmark_corr: {e}")
        return None

def information_ratio(df1, df2, tenor):
    try:
        if 'Target Return (%)' not in df1.columns or 'Target Return (%)' not in df2.columns:
            raise KeyError("Both DataFrames must contain a 'Target Return (%)' column.")
        if tenor <= 0 or tenor > len(df1) or tenor > len(df2):
            tenor = min(len(df1), len(df2))
        target_return1 = df1['Target Return (%)'].iloc[-tenor:].ffill().reset_index(drop=True)
        target_return2 = df2['Target Return (%)'].iloc[-tenor:].ffill().reset_index(drop=True)
        if len(target_return1) != len(target_return2):
            raise ValueError("Mismatch in lengths of the two series for the specified tenor.")
        active_return = target_return1 - target_return2
        tracking_error = np.std(active_return, ddof=1)
        if tracking_error == 0:
            return np.nan
        return np.mean(active_return) / tracking_error
    except Exception as e:
        print(f"An error occurred in information_ratio: {e}")
        return np.nan

def get_business_day_months_ago_in_data(data, tenor):
    end_date = data['Target Date'].max()
    start_date = end_date - relativedelta(months=tenor)
    if start_date < data['Target Date'].min():
        start_date = data['Target Date'].min()
    start_date = pd.Timestamp(start_date)
    if start_date.weekday() >= 5:
        start_date = pd.offsets.BDay().rollback(start_date)
    if (data['Target Date'] == start_date).any():
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)
    else:
        while not (data['Target Date'] == start_date).any():
            start_date -= pd.Timedelta(days=1)
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)

def get_business_day_years_ago_in_data(data, tenor):
    end_date = data['Target Date'].max()
    start_date = end_date - relativedelta(years=tenor)
    if start_date < data['Target Date'].min():
        start_date = data['Target Date'].min()
    start_date = pd.Timestamp(start_date)
    if start_date.weekday() >= 5:
        start_date = pd.offsets.BDay().rollback(start_date)
    if (data['Target Date'] == start_date).any():
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)
    else:
        while not (data['Target Date'] == start_date).any():
            start_date -= pd.Timedelta(days=1)
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)

def get_business_day_start_of_year_in_data(data):
    current_year = datetime.now().year
    start_date = pd.Timestamp(f'{current_year}-01-01')
    end_date = data['Target Date'].max()
    if start_date.weekday() >= 5:
        start_date = pd.offsets.BDay().rollback(start_date)
    if (data['Target Date'] == start_date).any():
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)
    else:
        while not (data['Target Date'] == start_date).any():
            start_date += pd.Timedelta(days=1)
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)

def calculate_inception_days(data):
    first_date = data['Target Date'].min()
    last_date = data['Target Date'].max()
    filtered_data = data[(data['Target Date'] >= first_date) & (data['Target Date'] <= last_date)]
    return len(filtered_data)

def process_tickers(tickers, benchmarks):
    benchmark1_prices = get_adj_close_prices(benchmarks[0])
    benchmark2_prices = get_adj_close_prices(benchmarks[1])
    if benchmark1_prices is not None:
        benchmark1_ln_returns = calculate_ln_returns(benchmark1_prices)
        benchmark1_return = [np.nan] + benchmark1_ln_returns
    if benchmark2_prices is not None:
        benchmark2_ln_returns = calculate_ln_returns(benchmark2_prices)
        benchmark2_return = [np.nan] + benchmark2_ln_returns

    benchmark1_df = benchmark1_prices[['Date', 'Close']].rename(
        columns={'Close': 'Target Price', 'Date': 'Target Date'}
    )
    benchmark1_df['Target Price'] = pd.Series(benchmark1_df['Target Price'].apply(force_scalar))
    benchmark1_df['Target Price'] = pd.to_numeric(benchmark1_df['Target Price'], errors='coerce')
    benchmark1_df['Target Return (%)'] = benchmark1_return

    benchmark2_df = benchmark2_prices[['Date', 'Close']].rename(
        columns={'Close': 'Target Price', 'Date': 'Target Date'}
    )
    benchmark2_df['Target Price'] = pd.Series(benchmark2_df['Target Price'].apply(force_scalar))
    benchmark2_df['Target Price'] = pd.to_numeric(benchmark2_df['Target Price'], errors='coerce')
    benchmark2_df['Target Return (%)'] = benchmark2_return

    results = {}
    for ticker in tickers:
        print(f"Processing {ticker}")
        prices = get_adj_close_prices(ticker)
        if prices is not None:
            ln_returns = calculate_ln_returns(prices)
            target_return = [np.nan] + ln_returns
            target_df = prices[['Date', 'Close']].copy()
            target_df.rename(columns={'Date': 'Target Date', 'Close': 'Target Price'}, inplace=True)
            target_df['Target Date'] = pd.to_datetime(target_df['Target Date'])
            target_df['Target Price'] = pd.Series(target_df['Target Price'].apply(force_scalar))
            target_df['Target Price'] = pd.to_numeric(target_df['Target Price'], errors='coerce')
            target_df['Target Return (%)'] = target_return
            target_df['Target Geodummy (%)'] = target_df['Target Return (%)'] + 100
            prices = add_target_drawdown(prices)
            target_df['Target Drawdown ($)'] = prices['Target Drawdown']
            prices = add_consecutive_drawdown_count(prices)
            target_df['Target Drawdown Recovery Length (Days)'] = prices['Consecutive Drawdown Count']
            prices = add_target_drawdown_ratio(prices)
            target_df['Target Drawdown Recovery Ratio (%)'] = prices['Target Drawdown Ratio']
            prices = add_target_price_ratio(prices)
            target_df['Target'] = prices['Target Price Ratio']
            
            days_1M = get_business_day_months_ago_in_data(target_df, 1)
            days_3M = get_business_day_months_ago_in_data(target_df, 3)
            days_6M = get_business_day_months_ago_in_data(target_df, 6)
            days_YTD = get_business_day_start_of_year_in_data(target_df)
            days_1Y = get_business_day_years_ago_in_data(target_df, 1)
            days_2Y = get_business_day_years_ago_in_data(target_df, 2)
            days_3Y = get_business_day_years_ago_in_data(target_df, 3)
            days_5Y = get_business_day_years_ago_in_data(target_df, 5)
            inception_days = calculate_inception_days(target_df)
            
            corr1_1M = calculate_benchmark_corr(target_df, benchmark1_df, get_business_day_months_ago_in_data(benchmark1_df, 1))
            corr1_YTD = calculate_benchmark_corr(target_df, benchmark1_df, get_business_day_start_of_year_in_data(benchmark1_df))
            corr1_1Y = calculate_benchmark_corr(target_df, benchmark1_df, get_business_day_years_ago_in_data(benchmark1_df, 1))
            corr1_2Y = calculate_benchmark_corr(target_df, benchmark1_df, get_business_day_years_ago_in_data(benchmark1_df, 2))
            corr1_3Y = calculate_benchmark_corr(target_df, benchmark1_df, get_business_day_years_ago_in_data(benchmark1_df, 3))
            corr1_5Y = calculate_benchmark_corr(target_df, benchmark1_df, get_business_day_years_ago_in_data(benchmark1_df, 5))
            corr1_inception = calculate_benchmark_corr(target_df, benchmark1_df, calculate_inception_days(benchmark1_df))
            
            corr2_1M = calculate_benchmark_corr(target_df, benchmark2_df, get_business_day_months_ago_in_data(benchmark2_df, 1))
            corr2_YTD = calculate_benchmark_corr(target_df, benchmark2_df, get_business_day_start_of_year_in_data(benchmark2_df))
            corr2_1Y = calculate_benchmark_corr(target_df, benchmark2_df, get_business_day_years_ago_in_data(benchmark2_df, 1))
            corr2_2Y = calculate_benchmark_corr(target_df, benchmark2_df, get_business_day_years_ago_in_data(benchmark2_df, 2))
            corr2_3Y = calculate_benchmark_corr(target_df, benchmark2_df, get_business_day_years_ago_in_data(benchmark2_df, 3))
            corr2_5Y = calculate_benchmark_corr(target_df, benchmark2_df, get_business_day_years_ago_in_data(benchmark2_df, 5))
            corr2_inception = calculate_benchmark_corr(target_df, benchmark2_df, calculate_inception_days(benchmark2_df))
            
            geomean_1M = calculate_geomean(target_df, days_1M)
            geomean_YTD = calculate_geomean(target_df, days_YTD)
            geomean_1Y = calculate_geomean(target_df, days_1Y)
            geomean_2Y = calculate_geomean(target_df, days_2Y)
            geomean_3Y = calculate_geomean(target_df, days_3Y)
            geomean_5Y = calculate_geomean(target_df, days_5Y-1)
            geomean_inception = calculate_geomean(target_df, inception_days-1)
            
            median_1M = calculate_median(target_df, days_1M)
            median_YTD = calculate_median(target_df, days_YTD)
            median_1Y = calculate_median(target_df, days_1Y)
            median_2Y = calculate_median(target_df, days_2Y)
            median_3Y = calculate_median(target_df, days_3Y)
            median_5Y = calculate_median(target_df, days_5Y)
            median_inception = calculate_median(target_df, inception_days)
            
            cfr_1M = calculate_cum_return(target_df, days_1M)
            cfr_YTD = calculate_cum_return(target_df, days_YTD)
            cfr_1Y = calculate_cum_return(target_df, days_1Y)
            cfr_2Y = calculate_cum_return(target_df, days_2Y)
            cfr_3Y = calculate_cum_return(target_df, days_3Y)
            cfr_5Y = calculate_cum_return(target_df, days_5Y-1)
            cfr_inception = calculate_cum_return(target_df, inception_days-1)
            
            std_1M = calculate_std(target_df, days_1M)
            if std_1M not in [None, "N/A"]:
                std_1M = std_1M * (days_1M ** 0.5) / 100
            std_YTD = calculate_std(target_df, days_YTD)
            if std_YTD not in [None, "N/A"]:
                std_YTD = std_YTD * (days_YTD ** 0.5) / 100
            std_1Y = calculate_std(target_df, days_1Y)
            if std_1Y not in [None, "N/A"]:
                std_1Y = std_1Y * (252 ** 0.5) / 100
            std_2Y = calculate_std(target_df, days_2Y)
            if std_2Y not in [None, "N/A"]:
                std_2Y = std_2Y * (252 ** 0.5) / 100
            std_3Y = calculate_std(target_df, days_3Y)
            if std_3Y not in [None, "N/A"]:
                std_3Y = std_3Y * (252 ** 0.5) / 100
            std_5Y = calculate_std(target_df, days_5Y)
            if std_5Y not in [None, "N/A"]:
                std_5Y = std_5Y * (252 ** 0.5) / 100
            std_inception = calculate_std(target_df, inception_days)
            if std_inception not in [None, "N/A"]:
                std_inception = std_inception * (252 ** 0.5) / 100
            
            kurt_1M = calculate_kurtosis(target_df, days_1M)
            kurt_YTD = calculate_kurtosis(target_df, days_YTD)
            kurt_1Y = calculate_kurtosis(target_df, days_1Y)
            kurt_2Y = calculate_kurtosis(target_df, days_2Y)
            kurt_3Y = calculate_kurtosis(target_df, days_3Y)
            kurt_5Y = calculate_kurtosis(target_df, days_5Y)
            kurt_inception = calculate_kurtosis(target_df, inception_days)
            
            skew_1M = calculate_skew(target_df, days_1M)
            skew_YTD = calculate_skew(target_df, days_YTD)
            skew_1Y = calculate_skew(target_df, days_1Y)
            skew_2Y = calculate_skew(target_df, days_2Y)
            skew_3Y = calculate_skew(target_df, days_3Y)
            skew_5Y = calculate_skew(target_df, days_5Y)
            skew_inception = calculate_skew(target_df, inception_days)
            
            min_1M = calculate_min_target_return(target_df, days_1M)
            min_YTD = calculate_min_target_return(target_df, days_YTD)
            min_1Y = calculate_min_target_return(target_df, days_1Y)
            min_2Y = calculate_min_target_return(target_df, days_2Y)
            min_3Y = calculate_min_target_return(target_df, days_3Y)
            min_5Y = calculate_min_target_return(target_df, days_5Y)
            min_inception = calculate_min_target_return(target_df, inception_days)
            
            max_1M = calculate_max_target_return(target_df, days_1M)
            max_YTD = calculate_max_target_return(target_df, days_YTD)
            max_1Y = calculate_max_target_return(target_df, days_1Y)
            max_2Y = calculate_max_target_return(target_df, days_2Y)
            max_3Y = calculate_max_target_return(target_df, days_3Y)
            max_5Y = calculate_max_target_return(target_df, days_5Y)
            max_inception = calculate_max_target_return(target_df, inception_days)
            
            calculated_results = pd.DataFrame()
            calculated_results.index = [
                f'Correlation with Benchmark #1 ({benchmarks[0]})',
                f'Correlation with Benchmark #2 ({benchmarks[1]})',
                'Geometric Mean of Fund Return (Annualized)',
                'Median Fund Return',
                'Cumulative Returns (Fund)',
                'Standard Deviation (Annualized)',
                'Kurtosis',
                'Skewness',
                'Minimum',
                'Maximum',
                'Average Drawdown Per Share',
                'Average Drawdown by Percentage',
                'Information Ratio',
                'Length of Drawdown',
                'Maximum Drawdown'
            ]
            
            calculated_results['YTD'] = [
                corr1_YTD, corr2_YTD, geomean_YTD, median_YTD, cfr_YTD,
                std_YTD, kurt_YTD, skew_YTD, min_YTD, max_YTD,
                avg_drawdown_YTD, avg_drawdown_perc_YTD, IR_YTD,
                len_drawdown_YTD, max_drawdown_YTD
            ]
            calculated_results['1 Month'] = [
                corr1_1M, corr2_1M, geomean_1M, median_1M, cfr_1M,
                std_1M, kurt_1M, skew_1M, min_1M, max_1M,
                avg_drawdown_1M, avg_drawdown_perc_1M, IR_1M,
                len_drawdown_1M, max_drawdown_1M
            ]
            calculated_results['1 Year'] = [
                corr1_1Y, corr2_1Y, geomean_1Y, median_1Y, cfr_1Y,
                std_1Y, kurt_1Y, skew_1Y, min_1Y, max_1Y,
                avg_drawdown_1Y, avg_drawdown_perc_1Y, IR_1Y,
                len_drawdown_1Y, max_drawdown_1Y
            ]
            calculated_results['2 Year'] = [
                corr1_2Y, corr2_2Y, geomean_2Y, median_2Y, cfr_2Y,
                None, kurt_2Y, None, min_2Y, max_2Y,
                avg_drawdown_2Y, avg_drawdown_perc_2Y, IR_2Y,
                len_drawdown_2Y, max_drawdown_2Y
            ]
            calculated_results['3 Year'] = [
                corr1_3Y, corr2_3Y, geomean_3Y, median_3Y, cfr_3Y,
                None, kurt_3Y, None, min_3Y, max_3Y,
                avg_drawdown_3Y, avg_drawdown_perc_3Y, IR_3Y,
                len_drawdown_3Y, max_drawdown_3Y
            ]
            calculated_results['5 Year'] = [
                corr1_5Y, corr2_5Y, geomean_5Y, median_5Y, cfr_5Y,
                std_5Y, kurt_5Y, skew_5Y, min_5Y, max_5Y,
                avg_drawdown_5Y, avg_drawdown_perc_5Y, IR_5Y,
                len_drawdown_5Y, max_drawdown_5Y
            ]
            calculated_results['Since Inception'] = [
                corr1_inception, corr2_inception, geomean_inception, median_inception,
                cfr_inception, std_inception, kurt_inception, skew_inception,
                min_inception, max_inception, avg_drawdown_inception,
                avg_drawdown_perc_inception, IR_inception,
                len_drawdown_inception, max_drawdown_inception
            ]
            
            results[ticker] = {"target": target_df, "calculated": calculated_results}
    return results

def main():
    tickers = input("Enter a list of tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOG): ").upper().split(', ')
    benchmarks = input("Enter two benchmark tickers separated by commas (e.g., ^SP500, ^DJI): ").upper().split(', ')
    
    if len(tickers) < 1:
        print("Please provide at least 1 ticker.")
        return

    results = process_tickers(tickers, benchmarks)
    
    output_filename = "combined_output.xlsx"
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        for ticker, sheets in results.items():
            df_target = sheets["target"]
            if isinstance(df_target.columns, pd.MultiIndex):
                df_target.columns = ['_'.join(map(str, col)).strip() for col in df_target.columns.values]
            df_calculated = sheets["calculated"]
            if isinstance(df_calculated.columns, pd.MultiIndex):
                df_calculated.columns = ['_'.join(map(str, col)).strip() for col in df_calculated.columns.values]
            df_target.to_excel(writer, sheet_name=f"{ticker}_Target", index=False)
            df_calculated.to_excel(writer, sheet_name=f"{ticker}_Calculated")
    print(f"Combined Excel file '{output_filename}' has been created with {len(results)*2} sheets.")

if __name__ == "__main__":
    main()
