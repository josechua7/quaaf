import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
import statistics
from scipy.stats import skew
from scipy.stats import kurtosis

def get_adj_close_prices(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if 'Close' in data.columns:
            adj_close_prices = data[['Close']]
            adj_close_prices.reset_index(inplace=True)
            return adj_close_prices
        else:
            raise ValueError("Adjusted Close prices not found in the data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_ln_returns(data):
    ln_returns = []
    close = data['Close'].values
    for i in range(1, len(close)):
        ln_return = np.log(close[i] / close[i - 1]) * 100
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

##------------------------##
def calculate_geomean(data, tenor):
    """
    Calculate geometric mean of return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the Target Geodummy (%) column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The minimum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Geodummy (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Geodummy (%)' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Geodummy (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return statistics.geometric_mean(observation_period)-100
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_median(data, tenor):
    """
    Calculate median fund return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'Target Return' column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The minimum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Return (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return statistics.median(observation_period)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def calculate_cum_return(data, tenor):
    """
    Calculate the percentage change in fund return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'Target Price' column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The percentage change in target return over the observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return (%)' column
        if 'Target Price' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Price' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            tenor = len(data)

        # Slice the data for the observation period
        observation_period = data['Target Price'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)

        # Calculate the percentage change
        first_value = observation_period.iloc[0].squeeze()
        last_value = observation_period.iloc[-1].squeeze()
        
        if first_value == 0:
            return "N/A"  # Avoid division by zero

        percentage_change = (last_value - first_value) / first_value

        return percentage_change
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_std(data, tenor):
    """
    Calculate sample standard deviation of return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'Target Return' column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The minimum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Return (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return observation_period.std(ddof=1)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_skew(data, tenor):
    """
    Calculate skewness of return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'Target Return' column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The minimum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Return (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return observation_period.skew()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_kurtosis(data, tenor):
    """
    Calculate kurtosis of return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'Target Return' column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The minimum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Return (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return observation_period.kurtosis()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_min_target_return(data, tenor):
    """
    Calculate the minimum target return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'Target Return' column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The minimum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Return (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return observation_period.min()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def calculate_max_target_return(data, tenor):
    """
    Calculate the maximum target return over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'Target Return' column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The maximum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Return (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Return' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Return (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return observation_period.max()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def calculate_avg_drawdown(data, tenor):
    """
    Calculate the average drawdown per share over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the Target Drawdown ($) column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The maximum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Drawdown ($)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown ($)' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Drawdown ($)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return statistics.mean(observation_period)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def calculate_avg_drawdown_percentage(data, tenor):
    """
    Calculate the average drawdown by percentage over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the Target Drawdown Recovery Ratio (%) column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The maximum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Drawdown Recovery Ratio (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown Recovery Ratio (%)' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Drawdown Recovery Ratio (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return statistics.mean(observation_period)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_drawdown_length(data, tenor):
    """
    Calculate the maximum length of drawdown over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the Target Drawdown Recovery Length (Days) column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The maximum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Drawdown Recovery Length (Days)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown Recovery Length (Days)' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Drawdown Recovery Length (Days)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return observation_period.max()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def calculate_max_drawdown(data, tenor):
    """
    Calculate the maximum drawdown (%) over a specified tenor.

    Parameters:
        data (pd.DataFrame): DataFrame containing the Target Drawdown Recovery Ratio (%) column.
        tenor (int): Number of periods to backtrack from the most recent observation.

    Returns:
        float: The maximum target return in the specified observation period.
    """
    try:
        # Ensure the DataFrame has the 'Target Return' column
        if 'Target Drawdown Recovery Ratio (%)' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Target Drawdown Recovery Ratio (%)' column.")
        
        # Ensure the tenor is valid
        if tenor <= 0 or tenor > len(data):
            return "N/A"

        # Slice the data for the observation period
        observation_period = data['Target Drawdown Recovery Ratio (%)'].iloc[-tenor:]
        
        observation_period = observation_period.reset_index(drop=True)
        
        # Calculate and return the minimum target return
        return observation_period.min()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def calculate_benchmark_corr(df1, df2, tenor):

    """
    Calculate the correlation between the 'Target Price' columns in two DataFrames over a specified tenor.

    Parameters:
        df1 (pd.DataFrame): The first DataFrame containing the 'Target Price' column.
        df2 (pd.DataFrame): The second DataFrame containing the 'Target Price' column.
        tenor (int): The number of most recent rows to consider for the correlation.

    Returns:
        float or str: The correlation coefficient, or "N/A" if conditions are invalid.
    """
    try:
        # Check if both DataFrames have the 'Target Price' column
        if 'Target Price' not in df1.columns or 'Target Price' not in df2.columns:
            raise KeyError("Both DataFrames must contain a 'Target Price' column.")

        # Ensure tenor is a positive integer and within the bounds of both DataFrames
        if tenor <= 0 or tenor > len(df1) or tenor > len(df2):
            tenor = min(len(df1), len(df2))
        
        # Extract the last `tenor` rows of the 'Target Price' column from each DataFrame
        target_price1 = df1['Target Price'].iloc[-tenor:].ffill().squeeze()
        target_price2 = df2['Target Price'].iloc[-tenor:].ffill().squeeze()
        
        target_price1 = target_price1.reset_index(drop=True)
        target_price2 = target_price2.reset_index(drop=True)
        
        # Ensure both slices have the same length (redundant, but safe)
        if len(target_price1) != len(target_price2):
            raise ValueError("Mismatch in lengths of the two series for the specified tenor.")

        # Calculate and return the correlation coefficient
        return target_price1.corr(target_price2)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def information_ratio(df1, df2, tenor):
    """
    Calculate the Information Ratio (IR) over the most recent 'tenor' days.

    Parameters:
    portfolio_returns (pd.Series, np.array, or pd.DataFrame): Portfolio returns
    benchmark_returns (pd.Series, np.array, or pd.DataFrame): Benchmark returns
    tenor (int, optional): Number of days before the most recent date. 
                           If None, computes IR over the entire dataset.

    Returns:
    float: Information Ratio value or NaN if calculation is not possible.
    """
    try:
        # Check if both DataFrames have the 'Target Price' column
        if 'Target Return (%)' not in df1.columns or 'Target Return (%)' not in df2.columns:
            raise KeyError("Both DataFrames must contain a 'Target Return (%)' column.")

        # Ensure tenor is a positive integer and within the bounds of both DataFrames
        if tenor <= 0 or tenor > len(df1) or tenor > len(df2):
            tenor = min(len(df1), len(df2))
        
        # Extract the last `tenor` rows of the 'Target Price' column from each DataFrame
        target_return1 = df1['Target Return (%)'].iloc[-tenor:].ffill().squeeze()
        target_return2 = df2['Target Return (%)'].iloc[-tenor:].ffill().squeeze()
        
        target_return1 = target_return1.reset_index(drop=True)
        target_return2 = target_return2.reset_index(drop=True)
        
        # Ensure both slices have the same length (redundant, but safe)
        if len(target_return1) != len(target_return2):
            raise ValueError("Mismatch in lengths of the two series for the specified tenor.")
            
        # Calculate active return (excess return)
        active_return = target_return1 - target_return2

        # Calculate tracking error (standard deviation of active return)
        tracking_error = np.std(active_return, ddof=1)  # Sample standard deviation

        # Prevent division by zero
        if tracking_error == 0:
            return np.nan

        # Calculate and return the Information Ratio
        return np.mean(active_return) / tracking_error
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.nan


##------------------------## # Day Count Functions
from dateutil.relativedelta import relativedelta

def get_business_day_months_ago_in_data(data, tenor):
    # Get the current date
    end_date = data['Target Date'].max()

    # Subtract x-months from the current date
    start_date = end_date - relativedelta(months=tenor)
    
    # Ensure the start date doesn't go before the minimum Target Date in the data
    if start_date < data['Target Date'].min():
        start_date = data['Target Date'].min()

    # Convert to a pandas Timestamp for easier handling
    start_date = pd.Timestamp(start_date)

    # Ensure the date is a business day, if it's not, adjust to the previous business day
    if start_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        start_date = pd.offsets.BDay().rollback(start_date)

    # Check if the date exists in data['Target Date']
    if start_date in data['Target Date'].values:
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)
    else:
        # If the date does not exist, rollback until we find a valid date
        while start_date not in data['Target Date'].values:
            start_date -= pd.Timedelta(days=1)  # Rollback one day
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)

def get_business_day_years_ago_in_data(data, tenor):
    # Get the current date
    end_date = data['Target Date'].max()

    # Subtract x-years from the current date
    start_date = end_date - relativedelta(years=tenor)
    
    # Ensure the start date doesn't go before the minimum Target Date in the data
    if start_date < data['Target Date'].min():
        start_date = data['Target Date'].min()

    # Convert to a pandas Timestamp for easier handling
    start_date = pd.Timestamp(start_date)

    # Ensure the date is a business day, if it's not, adjust to the previous business day
    if start_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        start_date = pd.offsets.BDay().rollback(start_date)

    # Check if the date exists in data['Target Date']
    if start_date in data['Target Date'].values:
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)
    else:
        # If the date does not exist, rollback until we find a valid date
        while start_date not in data['Target Date'].values:
            start_date -= pd.Timedelta(days=1)  # Rollback one day
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)

def get_business_day_start_of_year_in_data(data):
    # Get the current year
    current_year = datetime.now().year
    
    # Set the start_date to January 1st of the current year
    start_date = pd.Timestamp(f'{current_year}-01-01')

    # Get the most recent date from the data
    end_date = data['Target Date'].max()

    # Ensure the start_date is a business day, if it's not, adjust to the previous business day
    if start_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        start_date = pd.offsets.BDay().rollback(start_date)

    # Check if the start date exists in data['Target Date']
    if start_date in data['Target Date'].values:
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)
    else:
        # If the start date does not exist, rollback until we find a valid date
        while start_date not in data['Target Date'].values:
            start_date += pd.Timedelta(days=1)  # Rollback one day

        # Filter data based on the valid start date
        filtered_data = data[(data['Target Date'] >= start_date) & (data['Target Date'] <= end_date)]
        return len(filtered_data)

def calculate_inception_days(data):
    first_date = data['Target Date'].min()
    last_date = data['Target Date'].max()
    
    # Filter the data within the range of first and last date
    filtered_data = data[(data['Target Date'] >= first_date) & (data['Target Date'] <= last_date)]
    
    # Calculate the number of inception days
    inception_days = len(filtered_data)
    
    return inception_days



def process_tickers(tickers,benchmarks):

    benchmark1_prices = get_adj_close_prices(benchmarks[0])
    benchmark2_prices = get_adj_close_prices(benchmarks[1])

    if benchmark1_prices is not None:
        
            benchmark1_ln_returns = calculate_ln_returns(benchmark1_prices)
            benchmark1_return = [None] + [item for sublist in benchmark1_ln_returns for item in sublist]
    if benchmark2_prices is not None:
        
            benchmark2_ln_returns = calculate_ln_returns(benchmark2_prices)
            benchmark2_return = [None] + [item for sublist in benchmark2_ln_returns for item in sublist]        
    # benchmark1_prices = get_adj_close_prices(benchmarks[0])
    # benchmark2_prices = get_adj_close_prices(benchmarks[1])

    benchmark1_df = benchmark1_prices[['Date', 'Close']].rename(columns={'Close': 'Target Price', 'Date': 'Target Date'})
    benchmark1_df['Target Return (%)'] = benchmark1_return

    benchmark2_df = benchmark2_prices[['Date', 'Close']].rename(columns={'Close': 'Target Price', 'Date': 'Target Date'})
    benchmark2_df['Target Return (%)'] = benchmark2_return

    results = {}
    for ticker in tickers:
        print(f"Processing {ticker}")
        prices = get_adj_close_prices(ticker)
        if prices is not None:
            ln_returns = calculate_ln_returns(prices)
            target_return = [None] + [item for sublist in ln_returns for item in sublist]
            target_df = prices[['Date', 'Close']]
            target_df.rename(columns={'Date': 'Target Date', 'Close': 'Target Price'}, inplace=True)
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
            results[ticker] = target_df
            target_df.to_csv(f'{ticker}_target.csv')
        # (1M, Today) Days
        days_1M = get_business_day_months_ago_in_data(target_df, 1)
        days_1M_b1 = get_business_day_months_ago_in_data(benchmark1_df, 1)
        days_1M_b2 = get_business_day_months_ago_in_data(benchmark2_df, 1)

        # (3M, Today) Days
        days_3M = get_business_day_months_ago_in_data(target_df, 3)
        days_3M_b1 = get_business_day_months_ago_in_data(benchmark1_df, 3)
        days_3M_b2 = get_business_day_months_ago_in_data(benchmark2_df, 3)

        # (6M, Today) Days
        days_6M = get_business_day_months_ago_in_data(target_df, 6)
        days_6M_b1 = get_business_day_months_ago_in_data(benchmark1_df, 6)
        days_6M_b2 = get_business_day_months_ago_in_data(benchmark2_df, 6)

        # (YTD, Today) Days
        days_YTD = get_business_day_start_of_year_in_data(target_df)
        days_YTD_b1 = get_business_day_start_of_year_in_data(benchmark1_df)
        days_YTD_b2 = get_business_day_start_of_year_in_data(benchmark2_df)

        # (1Y, Today) Days
        days_1Y = get_business_day_years_ago_in_data(target_df, 1)
        days_1Y_b1 = get_business_day_years_ago_in_data(benchmark1_df, 1)
        days_1Y_b2 = get_business_day_years_ago_in_data(benchmark2_df, 1)

        # (2Y, Today) Days
        days_2Y = get_business_day_years_ago_in_data(target_df, 2)
        days_2Y_b1 = get_business_day_years_ago_in_data(benchmark1_df, 2)
        days_2Y_b2 = get_business_day_years_ago_in_data(benchmark2_df, 2)

        # (3Y, Today) Days
        days_3Y = get_business_day_years_ago_in_data(target_df, 3)
        days_3Y_b1 = get_business_day_years_ago_in_data(benchmark1_df, 3)
        days_3Y_b2 = get_business_day_years_ago_in_data(benchmark2_df, 3)

        # (3Y, Today) Days
        days_5Y = get_business_day_years_ago_in_data(target_df, 5)
        days_5Y_b1 = get_business_day_years_ago_in_data(benchmark1_df, 5)
        days_5Y_b2 = get_business_day_years_ago_in_data(benchmark2_df, 5)

        # (Inception, Today) Todays
        inception_days = calculate_inception_days(target_df)
        inception_days_b1 = calculate_inception_days(benchmark1_df)
        inception_days_b2 = calculate_inception_days(benchmark2_df)

        # Inception Day Count Calculation
        first_date = target_df['Target Date'].min()
        last_date = target_df['Target Date'].max()
        filtered_data = target_df[(target_df['Target Date'] >= first_date) & (target_df['Target Date'] <= last_date)]
        inception_days = len(filtered_data)


        # 1) Correlation with Benchmark1 Function
        corr1_1M = calculate_benchmark_corr(target_df, benchmark1_df, days_1M_b1)
        corr1_3M = calculate_benchmark_corr(target_df, benchmark1_df, days_3M_b1)
        corr1_6M = calculate_benchmark_corr(target_df, benchmark1_df, days_6M_b1)
        corr1_YTD = calculate_benchmark_corr(target_df, benchmark1_df, days_YTD_b1)
        corr1_1Y = calculate_benchmark_corr(target_df, benchmark1_df, days_1Y_b1)
        corr1_2Y = calculate_benchmark_corr(target_df, benchmark1_df, days_2Y_b1)
        corr1_3Y = calculate_benchmark_corr(target_df, benchmark1_df, days_3Y_b1)
        corr1_5Y = calculate_benchmark_corr(target_df, benchmark1_df, days_5Y_b1)
        corr1_inception = calculate_benchmark_corr(target_df, benchmark1_df, inception_days_b1)

        corr1 = [corr1_1M, corr1_3M, corr1_6M, corr1_YTD,
                corr1_1Y, corr1_2Y, corr1_3Y, corr1_5Y,
                corr1_inception]
        #print(corr1)

        # 2) Correlation with Benchmark2 Function
        corr2_1M = calculate_benchmark_corr(target_df, benchmark2_df, days_1M_b2)
        corr2_3M = calculate_benchmark_corr(target_df, benchmark2_df, days_3M_b2)
        corr2_6M = calculate_benchmark_corr(target_df, benchmark2_df, days_6M_b2)
        corr2_YTD = calculate_benchmark_corr(target_df, benchmark2_df, days_YTD_b2)
        corr2_1Y = calculate_benchmark_corr(target_df, benchmark2_df, days_1Y_b2)
        corr2_2Y = calculate_benchmark_corr(target_df, benchmark2_df, days_2Y_b2)
        corr2_3Y = calculate_benchmark_corr(target_df, benchmark2_df, days_3Y_b2)
        corr2_5Y = calculate_benchmark_corr(target_df, benchmark2_df, days_5Y_b2)
        corr2_inception = calculate_benchmark_corr(target_df, benchmark2_df, inception_days_b2)

        corr2 = [corr2_1M, corr2_3M, corr2_6M, corr2_YTD,
                corr2_1Y, corr2_2Y, corr2_3Y, corr2_5Y,
                corr2_inception]
        #print(corr2)

        # 3) Geometric Mean Return Function
        geomean_1M = calculate_geomean(target_df, days_1M)
        geomean_3M = calculate_geomean(target_df, days_3M)
        geomean_6M = calculate_geomean(target_df, days_6M)
        geomean_YTD = calculate_geomean(target_df, days_YTD)
        geomean_1Y = calculate_geomean(target_df, days_1Y)
        geomean_2Y = calculate_geomean(target_df, days_2Y)
        geomean_3Y = calculate_geomean(target_df, days_3Y)
        geomean_5Y = calculate_geomean(target_df, days_5Y-1)
        geomean_inception = calculate_geomean(target_df, inception_days-1)

        geomean_target = [geomean_1M, geomean_3M, geomean_6M, geomean_YTD,
                            geomean_1Y, geomean_2Y, geomean_3Y, geomean_5Y,
                            geomean_inception]
        #print(geomean_target)

        # 4) Median Fund Return Function
        median_1M = calculate_median(target_df, days_1M)
        median_3M = calculate_median(target_df, days_3M)
        median_6M = calculate_median(target_df, days_6M)
        median_YTD = calculate_median(target_df, days_YTD)
        median_1Y = calculate_median(target_df, days_1Y)
        median_2Y = calculate_median(target_df, days_2Y)
        median_3Y = calculate_median(target_df, days_3Y)
        median_5Y = calculate_median(target_df, days_5Y)
        median_inception = calculate_median(target_df, inception_days)

        median_target = [median_1M, median_3M, median_6M, median_YTD,
                            median_1Y, median_2Y, median_3Y, median_5Y,
                            median_inception]
        #print(median_target)

        # 5) Cumulative Fund Return Function
        cfr_1M = calculate_cum_return(target_df, days_1M)
        cfr_3M = calculate_cum_return(target_df, days_3M)
        cfr_6M = calculate_cum_return(target_df, days_6M)
        cfr_YTD = calculate_cum_return(target_df, days_YTD)
        cfr_1Y = calculate_cum_return(target_df, days_1Y)
        cfr_2Y = calculate_cum_return(target_df, days_2Y)
        cfr_3Y = calculate_cum_return(target_df, days_3Y)
        cfr_5Y = calculate_cum_return(target_df, days_5Y-1)
        cfr_inception = calculate_cum_return(target_df, inception_days-1)

        cfr_target = [cfr_1M, cfr_3M, cfr_6M, cfr_YTD,
                            cfr_1Y, cfr_2Y, cfr_3Y, cfr_5Y,
                            cfr_inception]
        #print(cfr_target)

        # 6) Call Standard Deviation (%) Function
        std_1M = calculate_std(target_df, days_1M)*(days_1M ** 0.5)/100
        std_3M = calculate_std(target_df, days_3M)*(days_3M ** 0.5)/100
        std_6M = calculate_std(target_df, days_6M)*(days_6M ** 0.5)/100
        std_YTD = calculate_std(target_df, days_YTD)*(days_YTD ** 0.5)/100
        std_1Y = calculate_std(target_df, days_1Y)*(252 ** 0.5)/100
        std_2Y = calculate_std(target_df, days_2Y)*(252 ** 0.5)/100
        std_3Y = calculate_std(target_df, days_3Y)*(252 ** 0.5)/100
        std_5Y = calculate_std(target_df, days_5Y)*(252 ** 0.5)/100
        std_inception = calculate_std(target_df, inception_days)*(252 ** 0.5)/100

        std_target = [std_1M, std_3M, std_6M, std_YTD,
                            std_1Y, std_2Y, std_3Y, std_5Y,
                            std_inception]
        #print(std_target)


        # 7) Call the Kurtosis Function
        kurt_1M = calculate_kurtosis(target_df, days_1M)
        kurt_3M = calculate_kurtosis(target_df, days_3M)
        kurt_6M = calculate_kurtosis(target_df, days_6M)
        kurt_YTD = calculate_kurtosis(target_df, days_YTD)
        kurt_1Y = calculate_kurtosis(target_df, days_1Y)
        kurt_2Y = calculate_kurtosis(target_df, days_2Y)
        kurt_3Y = calculate_kurtosis(target_df, days_3Y)
        kurt_5Y = calculate_kurtosis(target_df, days_5Y)
        kurt_inception = calculate_kurtosis(target_df, inception_days)

        kurt_target = [kurt_1M, kurt_3M, kurt_6M, kurt_YTD,
                            kurt_1Y, kurt_2Y, kurt_3Y, kurt_5Y,
                            kurt_inception]
        #print(kurt_target)

        # 8) Call the Skew Function
        skew_1M = calculate_skew(target_df, days_1M)
        skew_3M = calculate_skew(target_df, days_3M)
        skew_6M = calculate_skew(target_df, days_6M)
        skew_YTD = calculate_skew(target_df, days_YTD)
        skew_1Y = calculate_skew(target_df, days_1Y)
        skew_2Y = calculate_skew(target_df, days_2Y)
        skew_3Y = calculate_skew(target_df, days_3Y)
        skew_5Y = calculate_skew(target_df, days_5Y)
        skew_inception = calculate_skew(target_df, inception_days)

        skew_target = [skew_1M, skew_3M, skew_6M, skew_YTD,
                            skew_1Y, skew_2Y, skew_3Y, skew_5Y,
                            skew_inception]
        #print(skew_target)

        # 9) Call the Minimum Target Return Function
        min_1M = calculate_min_target_return(target_df, days_1M)
        min_3M = calculate_min_target_return(target_df, days_3M)
        min_6M = calculate_min_target_return(target_df, days_6M)
        min_YTD = calculate_min_target_return(target_df, days_YTD)
        min_1Y = calculate_min_target_return(target_df, days_1Y)
        min_2Y = calculate_min_target_return(target_df, days_2Y)
        min_3Y = calculate_min_target_return(target_df, days_3Y)
        min_5Y = calculate_min_target_return(target_df, days_5Y)
        min_inception = calculate_min_target_return(target_df, inception_days)

        min_return = [min_1M, min_3M, min_6M, min_YTD,
                            min_1Y, min_2Y, min_3Y, min_5Y,
                            min_inception]
        #print(min_return)

        # 10) Call the Minimum Target Return Function
        max_1M = calculate_max_target_return(target_df, days_1M)
        max_3M = calculate_max_target_return(target_df, days_3M)
        max_6M = calculate_max_target_return(target_df, days_6M)
        max_YTD = calculate_max_target_return(target_df, days_YTD)
        max_1Y = calculate_max_target_return(target_df, days_1Y)
        max_2Y = calculate_max_target_return(target_df, days_2Y)
        max_3Y = calculate_max_target_return(target_df, days_3Y)
        max_5Y = calculate_max_target_return(target_df, days_5Y)
        max_inception = calculate_max_target_return(target_df, inception_days)

        max_return = [max_1M, max_3M, max_6M, max_YTD,
                            max_1Y, max_2Y, max_3Y, max_5Y,
                            max_inception]
        #print(max_return)

        # 11) Average Drawdown per Share Function
        avg_drawdown_1M = calculate_avg_drawdown(target_df, days_1M)
        avg_drawdown_3M = calculate_avg_drawdown(target_df, days_3M)
        avg_drawdown_6M = calculate_avg_drawdown(target_df, days_6M)
        avg_drawdown_YTD = calculate_avg_drawdown(target_df, days_YTD)
        avg_drawdown_1Y = calculate_avg_drawdown(target_df, days_1Y)
        avg_drawdown_2Y = calculate_avg_drawdown(target_df, days_2Y)
        avg_drawdown_3Y = calculate_avg_drawdown(target_df, days_3Y)
        avg_drawdown_5Y = calculate_avg_drawdown(target_df, days_5Y)
        avg_drawdown_inception = calculate_avg_drawdown(target_df, inception_days)

        avg_drawdown = [avg_drawdown_1M, avg_drawdown_3M, avg_drawdown_6M, avg_drawdown_YTD,
                            avg_drawdown_1Y, avg_drawdown_2Y, avg_drawdown_3Y, avg_drawdown_5Y,
                            avg_drawdown_inception]
        #print(avg_drawdown)

        # 12) Average Drawdown by Percentage Function
        avg_drawdown_perc_1M = calculate_avg_drawdown_percentage(target_df, days_1M)
        avg_drawdown_perc_3M = calculate_avg_drawdown_percentage(target_df, days_3M)
        avg_drawdown_perc_6M = calculate_avg_drawdown_percentage(target_df, days_6M)
        avg_drawdown_perc_YTD = calculate_avg_drawdown_percentage(target_df, days_YTD)
        avg_drawdown_perc_1Y = calculate_avg_drawdown_percentage(target_df, days_1Y)
        avg_drawdown_perc_2Y = calculate_avg_drawdown_percentage(target_df, days_2Y)
        avg_drawdown_perc_3Y = calculate_avg_drawdown_percentage(target_df, days_3Y)
        avg_drawdown_perc_5Y = calculate_avg_drawdown_percentage(target_df, days_5Y)
        avg_drawdown_perc_inception = calculate_avg_drawdown_percentage(target_df, inception_days)

        avg_drawdown_perc = [avg_drawdown_perc_1M, avg_drawdown_perc_3M, avg_drawdown_perc_6M, avg_drawdown_perc_YTD,
                            avg_drawdown_perc_1Y, avg_drawdown_perc_2Y, avg_drawdown_perc_3Y, avg_drawdown_perc_5Y,
                            avg_drawdown_perc_inception]
        #print(avg_drawdown_perc)

        # 13) Information Ratio Function
        IR_1M = information_ratio(target_df, benchmark1_df, days_1M_b1)
        IR_3M = information_ratio(target_df, benchmark1_df, days_3M_b1)
        IR_6M = information_ratio(target_df, benchmark1_df, days_6M_b1)
        IR_YTD = information_ratio(target_df, benchmark1_df, days_YTD_b1)
        IR_1Y = information_ratio(target_df, benchmark1_df, days_1Y_b1)
        IR_2Y = information_ratio(target_df, benchmark1_df, days_2Y_b1)
        IR_3Y = information_ratio(target_df, benchmark1_df, days_3Y_b1)
        IR_5Y = information_ratio(target_df, benchmark1_df, days_5Y_b1)
        IR_inception = information_ratio(target_df, benchmark1_df, inception_days_b1)

        IR = [IR_1M, IR_3M, IR_6M, IR_YTD,
                IR_1Y, IR_2Y, IR_3Y, IR_5Y,
                IR_inception]
        #print(IR)

        # 14) Length of Drawdown Function
        len_drawdown_1M = calculate_drawdown_length(target_df, days_1M)
        len_drawdown_3M = calculate_drawdown_length(target_df, days_3M)
        len_drawdown_6M = calculate_drawdown_length(target_df, days_6M)
        len_drawdown_YTD = calculate_drawdown_length(target_df, days_YTD)
        len_drawdown_1Y = calculate_drawdown_length(target_df, days_1Y)
        len_drawdown_2Y = calculate_drawdown_length(target_df, days_2Y)
        len_drawdown_3Y = calculate_drawdown_length(target_df, days_3Y)
        len_drawdown_5Y = calculate_drawdown_length(target_df, days_5Y)
        len_drawdown_inception = calculate_drawdown_length(target_df, inception_days)

        len_drawdown = [len_drawdown_1M, len_drawdown_3M, len_drawdown_6M, len_drawdown_YTD,
                            len_drawdown_1Y, len_drawdown_2Y, len_drawdown_3Y, len_drawdown_5Y,
                            len_drawdown_inception]
        #print(len_drawdown)

        # 15) Maximum Drawdown Function
        max_drawdown_1M = calculate_max_drawdown(target_df, days_1M)
        max_drawdown_3M = calculate_max_drawdown(target_df, days_3M)
        max_drawdown_6M = calculate_max_drawdown(target_df, days_6M)
        max_drawdown_YTD = calculate_max_drawdown(target_df, days_YTD)
        max_drawdown_1Y = calculate_max_drawdown(target_df, days_1Y)
        max_drawdown_2Y = calculate_max_drawdown(target_df, days_2Y)
        max_drawdown_3Y = calculate_max_drawdown(target_df, days_3Y)
        max_drawdown_5Y = calculate_max_drawdown(target_df, days_5Y)
        max_drawdown_inception = calculate_max_drawdown(target_df, inception_days)

        max_drawdown = [max_drawdown_1M, max_drawdown_3M, max_drawdown_6M, max_drawdown_YTD,
                            max_drawdown_1Y, max_drawdown_2Y, max_drawdown_3Y, max_drawdown_5Y,
                            max_drawdown_inception]
        #print(max_drawdown)

        

        calculated_results = pd.DataFrame()
        calculated_results.index = [f'Correlation with Benchmark #1 ({benchmarks[0]})', f'Correlation with Benchmark #2 ({benchmarks[1]})', 
        'Geometric Mean of Fund Return (Annualized)','Median Fund Return','Cumulative Returns (Fund)',
        'Standard Deviation (Annualized)', 'Kurtosis','Skewness','Minimum','Maximum','Average Drawdown Per Share',
        'Average Drawdown by Percentage','Information Ratio','Length of Drawdown','Maximum Drawdown']

        calculated_results['YTD'] = [corr1_YTD, corr2_YTD, geomean_YTD, median_YTD, cfr_YTD, std_YTD, kurt_YTD, skew_YTD, min_YTD, 
                                    max_YTD, avg_drawdown_YTD, avg_drawdown_perc_YTD, IR_YTD, len_drawdown_YTD, max_drawdown_YTD]
        calculated_results['1 Month'] = [corr1_1M, corr2_1M, geomean_1M, median_1M, cfr_1M, std_1M, kurt_1M, skew_1M, min_1M, max_1M,
                                        avg_drawdown_1M, avg_drawdown_perc_1M, IR_1M, len_drawdown_1M, max_drawdown_1M]
        calculated_results['3 Month'] = [corr1_3M, corr2_3M, geomean_3M, median_3M, cfr_3M, std_3M, kurt_3M, skew_3M, min_3M, max_3M,
                                        avg_drawdown_3M, avg_drawdown_perc_3M, IR_3M, len_drawdown_3M, max_drawdown_3M]
        calculated_results['6 Month'] = [corr1_6M, corr2_6M, geomean_6M, median_6M, cfr_6M, std_6M, kurt_6M, skew_6M, min_6M, max_6M,
                                        avg_drawdown_6M, avg_drawdown_perc_6M, IR_6M, len_drawdown_6M, max_drawdown_6M]
        calculated_results['1 Year'] = [corr1_1Y, corr2_1Y, geomean_1Y, median_1Y, cfr_1Y, std_1Y, kurt_1Y, skew_1Y, min_1Y, max_1Y,
                                        avg_drawdown_1Y, avg_drawdown_perc_1Y, IR_1Y, len_drawdown_1Y, max_drawdown_1Y]
        calculated_results['2 Year'] = [corr1_2Y, corr2_2Y, geomean_2Y, median_2Y, cfr_2Y, std_2Y, kurt_2Y, skew_2Y, min_2Y, max_2Y,
                                        avg_drawdown_2Y, avg_drawdown_perc_2Y, IR_2Y, len_drawdown_2Y, max_drawdown_2Y]
        calculated_results['3 Year'] = [corr1_3Y, corr2_3Y, geomean_3Y, median_3Y, cfr_3Y, std_3Y, kurt_3Y, skew_3Y, min_3Y, max_3Y,
                                        avg_drawdown_3Y, avg_drawdown_perc_3Y, IR_3Y, len_drawdown_3Y, max_drawdown_3Y]
        calculated_results['5 Year'] = [corr1_5Y, corr2_5Y, geomean_5Y, median_5Y, cfr_5Y, std_5Y, kurt_5Y, skew_5Y, min_5Y, max_5Y,
                                        avg_drawdown_5Y, avg_drawdown_perc_5Y, IR_5Y, len_drawdown_5Y, max_drawdown_5Y]
        calculated_results['Since Inception'] = [corr1_inception, corr2_inception, geomean_inception, median_inception, 
                                                cfr_inception, std_inception, kurt_inception, skew_inception, 
                                                min_inception, max_inception, avg_drawdown_inception, avg_drawdown_perc_inception, 
                                                IR_inception, len_drawdown_inception, max_drawdown_inception]

        #print(f'{ticker} Calculated Results')
        #display(calculated_results)

        # Save as .csv
        calculated_results.to_csv(f'{ticker}_calculatedresults.csv')




    return results

def main():
    tickers = input("Enter a list of tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOG): ").upper().split(', ')
    benchmarks = input("Enter two benchmark tickers separated by commas (e.g., ^SP500, ^DJI): ").upper().split(', ')

    if len(tickers) < 1:
        print("Please provide at least 4 tickers (1 target, 3 comparables).")
        return
    results = process_tickers(tickers,benchmarks)
    for ticker, df in results.items():
        print(f"\n{ticker} Target Data:")
        print(df.head())

if __name__ == "__main__":
    main()
