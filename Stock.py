import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Get company ticker symbol
company = input("Enter a company ticker (e.g., AAPL, TSLA, GOOGL): ")

# Download real-time stock data
print(f"Downloading data for {company}...")
df = yf.download(company, period="2y", progress=False)

if df.empty:
    print(f"No data found for {company}. Please check the ticker symbol.")
    exit()

print(f"Downloaded {len(df)} days of data")

# Reset index to make Date a column
df.reset_index(inplace=True)

# Check data types
print("\nData types:")
print(df.dtypes)
print(f"\nData shape: {df.shape}")

# Drop 'Adj Close' if it exists
if 'Adj Close' in df.columns:
    df.drop('Adj Close', axis=1, inplace=True)

# Ensure Volume is float
df['Volume'] = df['Volume'].astype(float)

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Remove any rows with infinite values
df_new = df[np.isfinite(df.select_dtypes(include=[np.number])).all(1)].copy()

# Plot opening price with actual dates
plt.figure(figsize=(16, 6))
plt.plot(df_new['Date'], df_new['Open'], linewidth=1.5)
plt.title(f'{company} Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

# Prepare features and target
x = df_new[['Open', 'High', 'Low', 'Volume']].values
y = df_new['Close'].values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print("\nModel Coefficients:")
print(regressor.coef_)
print(f"\nModel Intercept: {regressor.intercept_}")

# Make predictions
predicted = regressor.predict(x_test)

# Create comparison dataframe - FIX: ensure 1D arrays
dfr = pd.DataFrame({
    'Actual': y_test.flatten() if len(y_test.shape) > 1 else y_test,
    'Predicted': predicted.flatten() if len(predicted.shape) > 1 else predicted
})

print("\nPrediction Results (first 10):")
print(dfr.head(10))

# Plot actual vs predicted
plt.figure(figsize=(16, 6))
indices = range(len(dfr))

plt.plot(indices, dfr['Actual'].values, color='blue', label='Actual', linewidth=2, marker='o', markersize=3)
plt.plot(indices, dfr['Predicted'].values, color='red', label='Predicted', linewidth=2, marker='x', markersize=3, alpha=0.7)
plt.xlabel("Test Sample Index")
plt.ylabel("Stock Closing Price ($)")
plt.title(f"{company} - Actual vs Predicted Closing Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Scatter plot for better visualization
plt.figure(figsize=(10, 10))
plt.scatter(dfr['Actual'], dfr['Predicted'], alpha=0.5)
plt.plot([dfr['Actual'].min(), dfr['Actual'].max()], 
         [dfr['Actual'].min(), dfr['Actual'].max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'{company} - Prediction Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Model performance
score = regressor.score(x_test, y_test)
print(f"\nModel R² Score: {score:.4f}")
print(f"(1.0 = perfect prediction, 0.0 = poor prediction)")

# Calculate additional metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predicted)
mape = np.mean(np.abs((y_test - predicted) / y_test)) * 100

print(f"\nMean Squared Error: ${mse:.4f}")
print(f"Root Mean Squared Error: ${rmse:.4f}")
print(f"Mean Absolute Error: ${mae:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")


#Get the latest live price
print("\n--- Current Market Data ---")
ticker = yf.Ticker(company)
current_data = ticker.history(period="1d")
if not current_data.empty:
    current_price = current_data['Close'].iloc[-1]
    print(f"Current market price: ${current_price:.2f}")
    print(f"Market status: {'Open' if datetime.now().weekday() < 5 else 'Closed (Weekend)'}")
else:
    print("Market is currently closed or data unavailable")

# Show last 5 days of actual data
print("\n--- Last 5 Days of Data ---")
print(df_new[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail())
