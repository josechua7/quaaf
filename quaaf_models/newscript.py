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

import pandas as pd

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



def process_tickers(tickers):
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
    return results

def main():
    tickers = input("Enter a list of tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOG): ").upper().split(', ')
    if len(tickers) < 4:
        print("Please provide at least 4 tickers (1 target, 3 comparables).")
        return
    results = process_tickers(tickers)
    for ticker, df in results.items():
        print(f"\n{ticker} Target Data:")
        print(df.head())

if __name__ == "__main__":
    main()
