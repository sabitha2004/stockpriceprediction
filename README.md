import yfinance as yf

# Define the stock symbol and date range
stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2021-12-31"

# Fetch historical stock price data
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Drop missing values if any
data = data.dropna()

# Select only the 'Close' price as the target variable
data = data[['Close']]

# Example: Calculate 50-day and 200-day moving averages
data['50MA'] = data['Close'].rolling(window=50).mean()
data['200MA'] = data['Close'].rolling(window=200).mean()

# Define the proportion of data to use for training
train_size = 0.8
split_index = int(len(data) * train_size)

# Split the data into training and testing sets
train_data = data[:split_index]
test_data = data[split_index:]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create and train the model
model = LinearRegression()
model.fit(train_data[['50MA', '200MA']], train_data['Close'])

# Make predictions
predictions = model.predict(test_data[['50MA', '200MA']])

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(test_data['Close'], predictions)
print(f"Mean Squared Error: {mse}")

import matplotlib.pyplot as plt

# Plot actual vs. predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Close'], label='Actual Stock Price', color='blue')
plt.plot(test_data.index, predictions, label='Predicted Stock Price', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.legend()
plt.show()


