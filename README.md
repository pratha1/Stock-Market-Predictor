# Stock Price Prediction using Linear Regression

A Python-based machine learning project that predicts stock closing prices using real-time data from Yahoo Finance. The model uses Linear Regression to analyze historical stock data and make predictions based on opening price, high, low, and trading volume.

## Features

- **Real-time Data Fetching**: Automatically downloads up to 2 years of historical stock data using yfinance
- **Machine Learning Prediction**: Uses scikit-learn's Linear Regression model to predict closing prices
- **Visual Analytics**: Generates multiple plots to visualize price trends and prediction accuracy
- **Performance Metrics**: Provides comprehensive model evaluation with R², RMSE, MAE, and MAPE
- **Live Market Data**: Fetches current market prices for comparison

## Prerequisites

Before running this project, ensure you have Python 3.7+ installed on your system.

## Installation

1. Clone or download this repository to your local machine

2. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib yfinance
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
yfinance>=0.2.0
```

## Usage
1. Navigate to the project directory:
```bash
cd path/to/stockPredictionProject
```

2. Run the script:
```bash
python Final_stockcode.py
```

3. When prompted, enter a valid stock ticker symbol:
```
Enter a company ticker (e.g., AAPL, TSLA, GOOGL): AAPL
```

### Popular Stock Ticker Examples

- **AAPL** - Apple Inc.
- **TSLA** - Tesla, Inc.
- **GOOGL** - Alphabet Inc. (Google)
- **MSFT** - Microsoft Corporation
- **AMZN** - Amazon.com, Inc.
- **META** - Meta Platforms, Inc. (Facebook)
- **NVDA** - NVIDIA Corporation
- **JPM** - JPMorgan Chase & Co.

## How It Works

1. **Data Collection**: Downloads 2 years of historical stock data from Yahoo Finance
2. **Data Preprocessing**: 
   - Removes missing or infinite values
   - Converts data types for consistency
   - Drops adjusted close prices to avoid redundancy
3. **Feature Selection**: Uses Open, High, Low, and Volume as features to predict Close price
4. **Model Training**: Splits data into 80% training and 20% testing sets
5. **Prediction**: Makes predictions on test data and evaluates performance
6. **Visualization**: Generates three key plots:
   - Historical opening price trend
   - Actual vs Predicted closing prices comparison
   - Scatter plot showing prediction accuracy

## Output

The script provides:

### Console Output
- Data download confirmation
- Missing value analysis
- Model coefficients and intercept
- First 10 prediction results
- Performance metrics (R², MSE, RMSE, MAE, MAPE)
- Current market price
- Last 5 days of historical data

### Visual Output
- **Figure 1**: Opening price trend over time
- **Figure 2**: Line plot comparing actual vs predicted prices
- **Figure 3**: Scatter plot showing prediction accuracy against perfect prediction line

## Future Improvements

- Add technical indicators (RSI, MACD, Moving Averages)
- Implement more advanced models (Random Forest, LSTM, XGBoost)
- Include sentiment analysis from news/social media
- Add future price prediction capabilities
- Implement cross-validation for better model evaluation
- Create a web interface for easier interaction

# Disclaimer

**This project is for educational purposes only.** The predictions made by this model should NOT be used as financial advice or for making actual investment decisions. Stock market investments carry risk, and past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Acknowledgments

- **yfinance** library for providing easy access to Yahoo Finance data
- **scikit-learn** for machine learning tools
- **pandas** and **numpy** for data manipulation
- **matplotlib** for data visualization

**Happy Predicting! 📈**
