#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:22:34 2024

@author: pavelkim
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Load the dataset
df = pd.read_csv('desktop/commodity.csv')

# Check the column names to ensure they match
print(df.columns)

# Rename columns to match the expected names if necessary
df.rename(columns={'Symbol': 'Commodity', 'Data': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Data cleaning: handle missing values by forward filling
df.fillna(method='ffill', inplace=True)

# Exploratory Data Analysis
def plot_commodity_prices(df, commodity):
    plt.figure(figsize=(14, 7))
    plt.plot(df[df['Commodity'] == commodity]['Close'])
    plt.title(f'{commodity} Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Visualize price trends for each commodity
commodities = df['Commodity'].unique()
for commodity in commodities:
    plot_commodity_prices(df, commodity)

# Feature Engineering: Calculate moving averages
df['MA50'] = df.groupby('Commodity')['Close'].transform(lambda x: x.rolling(window=50).mean())
df['MA200'] = df.groupby('Commodity')['Close'].transform(lambda x: x.rolling(window=200).mean())

# Additional Visualizations: Moving Averages
def plot_moving_averages(df, commodity):
    plt.figure(figsize=(14, 7))
    commodity_data = df[df['Commodity'] == commodity]
    plt.plot(commodity_data['Close'], label='Close Price')
    plt.plot(commodity_data['MA50'], label='50-Day MA')
    plt.plot(commodity_data['MA200'], label='200-Day MA')
    plt.title(f'{commodity} Prices with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Visualize moving averages for each commodity
for commodity in commodities:
    plot_moving_averages(df, commodity)

# Additional Visualizations: Heatmap of correlations
def plot_correlation_heatmap(df):
    plt.figure(figsize=(14, 7))
    corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Plot the correlation heatmap
plot_correlation_heatmap(df)

# Additional Visualizations: Bar chart for average prices
def plot_average_prices(df):
    plt.figure(figsize=(14, 7))
    avg_prices = df.groupby('Commodity')['Close'].mean().sort_values()
    avg_prices.plot(kind='bar')
    plt.title('Average Closing Prices by Commodity')
    plt.xlabel('Commodity')
    plt.ylabel('Average Closing Price')
    plt.show()

# Plot the average closing prices by commodity
plot_average_prices(df)

# Additional Visualizations: Histogram of price distributions
def plot_price_histograms(df):
    commodities = df['Commodity'].unique()
    num_commodities = len(commodities)
    fig, axes = plt.subplots(nrows=num_commodities, ncols=1, figsize=(14, 2 * num_commodities))
    fig.tight_layout(pad=5.0)
    
    for ax, commodity in zip(axes, commodities):
        commodity_data = df[df['Commodity'] == commodity]['Close']
        sns.histplot(commodity_data, kde=True, ax=ax, bins=50)
        ax.set_title(f'Price Distribution of {commodity}')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
    
    plt.show()

# Plot the histograms of price distributions
plot_price_histograms(df)

# Predictive Modeling: ARIMA
def arima_model(df, commodity, end_date='2030-01-01'):
    data = df[df['Commodity'] == commodity]['Close']
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:len(data)]
    
    # Fit the ARIMA model
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.forecast(steps=len(test))
    future_steps = (pd.to_datetime(end_date) - data.index[-1]).days
    future_predictions = model_fit.forecast(steps=future_steps)
    
    # Evaluate model performance
    error = mean_squared_error(test, predictions)
    print(f'{commodity} - Test MSE: {error}')
    
    # Plot predictions
    plt.figure(figsize=(14, 7))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(test.index, predictions, color='red', label='Predictions')
    plt.title(f'{commodity} Price Prediction using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Plot future predictions
    plt.figure(figsize=(14, 7))
    plt.plot(data, label='Historical Data')
    plt.plot(pd.date_range(start=data.index[-1], periods=future_steps, freq='D'), future_predictions, color='orange', label='Future Predictions')
    plt.title(f'{commodity} Future Price Prediction using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Apply ARIMA model to each commodity and predict until 2030
for commodity in commodities:
    arima_model(df, commodity, end_date='2030-01-01')
