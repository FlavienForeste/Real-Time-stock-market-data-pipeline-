from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("StockMarketAnalysis") \
    .getOrCreate()

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data.reset_index()

# Fetch some sample data
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

all_data = pd.concat([fetch_stock_data(symbol, start_date, end_date) for symbol in symbols])

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(all_data)

# Data Cleaning and Transformation
cleaned_df = spark_df.withColumn("Date", to_date("Date")) \
    .withColumn("Symbol", lit("AAPL"))  # Add this column as we're using multiple stocks

# Calculate 5-day moving average
window_spec = Window.partitionBy("Symbol").orderBy("Date").rowsBetween(-4, 0)
moving_avg_df = cleaned_df.withColumn("MA5", avg("Close").over(window_spec))

# Calculate daily returns
returns_df = moving_avg_df.withColumn("DailyReturn", 
    (col("Close") - lag("Close", 1).over(Window.partitionBy("Symbol").orderBy("Date"))) / lag("Close", 1).over(Window.partitionBy("Symbol").orderBy("Date")))

# Calculate volatility (standard deviation of returns)
volatility_df = returns_df.groupBy("Symbol") \
    .agg(stddev("DailyReturn").alias("Volatility"))

# Join the results
final_df = returns_df.join(volatility_df, "Symbol")

# Show the results
final_df.show()

# Save the results
final_df.write.mode("overwrite").parquet("stock_analysis_results")

# Stop the Spark session
spark.stop()