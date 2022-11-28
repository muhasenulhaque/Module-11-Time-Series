# Module-11-Time-Series
ML on Time Series Data like Prophet 
## Install and import the required libraries and dependencies
!pip install pystan
!pip install hvplot
!pip install holoviews
!pip install prophet

# Import the required libraries and dependencies

import pandas as pd
import holoviews as hv
from prophet import Prophet
from pathlib import Path
import hvplot.pandas
import datetime as dt
import numpy as np

%matplotlib inline

# Upload the "google_hourly_search_trends.csv" file into Colab, then store in a Pandas DataFrame

from google.colab import files
uploaded = files.upload()

# Set the "Date" column as the Datetime Index.

df_mercado_trends = pd.read_csv(
    "google_hourly_search_trends.csv",
    index_col='Date',
    parse_dates=True,
    infer_datetime_format=True
).dropna()

# Review the first and last five rows of the DataFrame
display(df_mercado_trends.head())
display(df_mercado_trends.tail())

# Holoviews extension to render hvPlots in Colab
hv.extension("bokeh")


# Slice the DataFrame to just the month of May 2020
df_may_2020 = df_mercado_trends["Search Trends"].loc["2020-05"]

# Plot the DataFrame
df_may_2020.hvplot()

# Calculate the sum of the total search traffic for May 2020
traffic_may_2020 = df_may_2020.sum()

# View the traffic_may_2020 value
traffic_may_2020

# Calcluate the monhtly median search traffic across all months 
# Group the DataFrame by index year and then index month, chain the sum and then the median functions
sum_monthly_traffic = df_mercado_trends["Search Trends"].groupby(by=[df_mercado_trends.index.year, df_mercado_trends.index.month]).sum()
median_monthly_traffic = df_mercado_trends["Search Trends"].groupby(by=[df_mercado_trends.index.year, df_mercado_trends.index.month]).median()

# View the median_monthly_traffic value
median_monthly_traffic
# Compare the seach traffic for the month of May 2020 to the overall monthly median value
df_median = df_mercado_trends["Search Trends"].median()
df_May2020_median = df_may_2020.median()
display(print(f"The overall median traffic {df_median}"))
display(print(f"The median traffic for the month of may {df_May2020_median}"))

Question: Did the Google search traffic increase during the month that MercadoLibre released its financial results?

Answer: Yes. The google search increased during during the month (May 2020) that MercadoLibre released its financial results.

## Step 2: Mine the Search Traffic Data for Seasonality
#### Step 1: Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Group the hourly search data to plot (use hvPlot) the average traffic by the day of week 
df_mercado_search_trends = df_mercado_trends['Search Trends']
group_level = df_mercado_trends.index.dayofweek
df_mercado_search_trends.groupby(group_level).mean().hvplot()

#### Step 2: Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?
# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the hour of the day and day of week search traffic as a heatmap.
df_mercado_trends.hvplot.heatmap(x='index.hour', y='index.dayofweek', C='Search Trends', cmap='reds').aggregate(function=np.mean)

Does any day-of-week effect that you observe concentrate in just a few hours of that day? Anser: The concentration of search is the in the early hours and the late hous of the day. Hours 5 to 11 the frequency of the seach comparatively very low.

Day 1 has got higer seach in first 4 hours compared to others.

#### Step 3: Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Group the hourly search data to plot (use hvPlot) the average traffic by the week of the year
df_mercado_search_weekly_trends = df_mercado_trends['Search Trends']
group_level_week = df_mercado_search_weekly_trends.index.weekofyear
df_mercado_search_weekly_trends.groupby(group_level_week).mean().hvplot()

Question: Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

Anser: There is a increase during the week 44 to 50 due to winter

## Step 3: Relate the Search Traffic to Stock Price Patterns
#### Step 1: Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

 #Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame

from google.colab import files
uploaded = files.upload()


# Store in a Pandas DataFrame and set the "date" column as the Datetime Index.
df_mercado_stock = pd.read_csv(
    "mercado_stock_price.csv",
    index_col='date',
    parse_dates=True,
    infer_datetime_format=True,
).dropna()


# View the first and last five rows of the DataFrame
display(df_mercado_stock.head())
display(df_mercado_stock.tail())

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the closing price of the df_mercado_stock DataFrame
df_mercado_stock.hvplot()


from pandas.core.generic import Axis
# Concatenate the df_mercado_stock DataFrame with the df_mercado_trends DataFrame
# Concatenate the DataFrame by columns (axis=1), and drop and rows with only one column of data
mercado_stock_trends_df = pd.concat([df_mercado_stock,df_mercado_trends],axis =1, join='inner')

# View the first and last five rows of the DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())

# For the combined dataframe, slice to just the first half of 2020 (2020-01 through 2020-06) 
first_half_2020 = mercado_stock_trends_df.loc["2020-01":"2020-06"]

# View the first and last five rows of first_half_2020 DataFrame
display(first_half_2020.head())
display(first_half_2020.tail())

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the close and Search Trends data
# Plot each column on a separate axes using the following syntax
# `hvplot(shared_axes=False, subplots=True).cols(1)`
first_half_2020.hvplot(shared_axes=False, subplots=True).cols(1)

**Question:** Do both time series indicate a common trend that’s consistent with this narrative?

**Answer:** The closing price and search trend does not follow the same trand during the first half (January to June)

#### Step 3: Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:

# Create a new column in the mercado_stock_trends_df DataFrame called Lagged Search Trends
# This column should shift the Search Trends information by one hour
mercado_stock_trends_df["Lagged Search Trends"] = mercado_stock_trends_df["Search Trends"].shift(1)

# Create a new column in the mercado_stock_trends_df DataFrame called Stock Volatility
# This column should calculate the standard deviation of the closing stock price return data over a 4 period rolling window
mercado_stock_trends_df['Stock Volatility'] = mercado_stock_trends_df['close'].rolling(4).std()

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the stock volatility
mercado_stock_trends_df['Stock Volatility'].hvplot()

# Create a new column in the mercado_stock_trends_df DataFrame called Hourly Stock Return
# This column should calculate hourly return percentage of the closing price
mercado_stock_trends_df['Hourly Stock Return'] = mercado_stock_trends_df['close'].pct_change()

# View the first and last five rows of the mercado_stock_trends_df DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())

#### Step 4: Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

# Construct correlation table of Stock Volatility, Lagged Search Trends, and Hourly Stock Return
mercado_stock_trends_df[["Stock Volatility","Lagged Search Trends","Hourly Stock Return"]].corr()

**Question:** Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

**Answer:** The correlation between Lagged Search Trends and Stock Volatility	and Hourly Stock Return is not stong enough to preduct the stock price or hourly stock return based on the rearch data. The corelation fugres are -0.118945	with Stock Volatility	and 0.017929 with Hourly Stock Return. The numbers suggest very weak or no correlation with stock bovality and stock return. 
## Step 4: Create a Time Series Model with Prophet

#### Step 1: Set up the Google search data for a Prophet forecasting model.

# Using the df_mercado_trends DataFrame, reset the index so the date information is no longer the index
mercado_prophet_df = df_mercado_trends.reset_index()

# Review the first and last five rows of the DataFrame
display(mercado_prophet_df.head())
display(mercado_prophet_df.tail())

# Label the columns ds and y so that the syntax is recognized by Prophet
mercado_prophet_df.columns = ['ds', 'y']
mercado_prophet_df.head()

# Drop an NaN values from the prophet_df DataFrame
mercado_prophet_df = mercado_prophet_df.dropna()
# View the first and last five rows of the mercado_prophet_df DataFrame

mercado_prophet_df.tail()

# Call the Prophet function, store as an object
model_mercado_trends = Prophet()
model_mercado_trends

# Fit the Prophet model.
model_mercado_trends.fit(mercado_prophet_df)

# Create a future dataframe to hold predictions
# Make the prediction go out as far as 2000 hours (approx 80 days)
future_mercado_trends = model_mercado_trends.make_future_dataframe(periods=2000, freq="H")

# View the last five rows of the future_mercado_trends DataFrame
future_mercado_trends.tail()

# Make the predictions for the trend data using the future_trends DataFrame
forecast_mercado_trends = model_mercado_trends.predict(future_mercado_trends)

# Display the first five rows of the forecast DataFrame
forecast_mercado_trends.head()

# Plot the Prophet predictions for the Mercado trends data
model_mercado_trends.plot(forecast_mercado_trends)

# Use the plot_components function to visualize the forecast results 
figures = model_mercado_trends.plot_components(forecast_mercado_trends)

**Question:** What time of day exhibits the greatest popularity?

**Answer:**  around 20:30 pm around 2 pm 

**Question:** Which day of week gets the most search traffic? 

**Answer:** Most Traffic is on Tuesday

**Question:** What's the lowest point for search traffic in the calendar year?

**Answer:** Novemver is the lowest point for search traffic. 

## Step 5 (Optional): Forecast Revenue by Using Time Series Models

# Upload the "mercado_daily_revenue.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the DatetimeIndex
# Sales are quoted in millions of US dollars
from google.colab import files
uploaded = files.upload()


df_mercado_sales = pd.read_csv(
    "mercado_daily_revenue.csv",
    index_col='date',
    parse_dates=True,
    infer_datetime_format=True,
).dropna()

# Review the DataFrame
df_mercado_sales.head()

# Holoviews extension to render hvPlots in Colab
hv.extension('bokeh')

# Use hvPlot to visualize the daily sales figures 
df_mercado_sales.hvplot()

# Apply a Facebook Prophet model to the data.

# Set up the dataframe in the neccessary format:
# Reset the index so that date becomes a column in the DataFrame
mercado_sales_prophet_df = df_mercado_sales.reset_index()


# Adjust the columns names to the Prophet syntax
mercado_sales_prophet_df.columns = ['ds', 'y']

# Visualize the DataFrame
mercado_sales_prophet_df.tail()

# Create the model
mercado_sales_prophet_model = Prophet()

# Fit the model
mercado_sales_prophet_model.fit(mercado_sales_prophet_df)

# Predict sales for 90 days (1 quarter) out into the future.

# Start by making a future dataframe
mercado_sales_prophet_future = mercado_sales_prophet_model.make_future_dataframe(periods=90, freq="D")

# Display the last five rows of the future DataFrame
mercado_sales_prophet_future.tail()

# Make predictions for the sales each day over the next quarter
mercado_sales_prophet_forecast = mercado_sales_prophet_model.predict(mercado_sales_prophet_future)

# Display the first 5 rows of the resulting DataFrame
mercado_sales_prophet_forecast.head()

# Use the plot_components function to analyze seasonal patterns in the company's revenue
mercado_sales_prophet_model.plot_components(mercado_sales_prophet_forecast)

**Question:** For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

**Answer:** # Wednesday
