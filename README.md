# Stock Price Prediction Analysis using Prophet and LSTM models

## Overview
The **Amazon Stock Price Prediction Analysis** project leverages data science and machine learning to analyze and predict Amazon stock prices. Using historical data, we apply both Prophet and LSTM models to generate reliable forecasts, providing insights into stock trends and prediction performance.

### Objectives
- Analyze Amazon stock data to uncover key trends and insights.
- Apply time-series models (Prophet and LSTM) for accurate stock price predictions.
- Compare prediction performances across models.
- Visualize historical data, forecasted trends, and key insights.

## Features
- **Data Retrieval**:
  - Fetch historical stock data using Yahoo Finance API.
  - Retrieve closing prices, trading volumes, and other related metrics.
- **Exploratory Data Analysis (EDA)**:
  - Generate visualizations for trends, seasonality, and stock price distributions.
  - Identify relationships between trading volumes and stock prices.
- **Prophet Model**:
  - Time-series forecasting with decomposed trend and seasonality components.
  - Generate confidence intervals for predictions (yhat_lower, yhat_upper).
- **LSTM Model**:
  - Sequence-to-sequence modeling for precise stock price prediction.
  - Sliding window sequences for input and one-day predictions.
  - MinMax scaling to normalize the price data.
- **Comparison**:
  - Compare observed vs. predicted prices across both models for evaluation.

## Technologies Used
- **Programming Language**: Python 3.6
- **Key Libraries**:
  - **Data Manipulation**: Pandas, NumPy
  - **Visualization**: Matplotlib, Plotly Express
  - **Machine Learning**: Prophet, TensorFlow/Keras (LSTM)
  - **APIs**: Yahoo Finance API
  - **Notebook Environment**: Jupyter Notebook

## Installation
1. Clone the repository: git clone https://github.com/shalini363/amazon-stock-price-prediction.git
   
2. Navigate to the project directory: cd amazon-stock-price-prediction
 
3. Install the required dependencies: pip install -r requirements.txt
 

## Usage
1. Open the Jupyter Notebook:
  
   jupyter notebook amazon_price_prediction_analysis.ipynb
  
2. Follow the notebook's step-by-step guide to:
   - Retrieve data from Yahoo Finance.
   - Conduct EDA and prepare data for modeling.
   - Train and evaluate the Prophet and LSTM models.
3. Customize the notebook to test predictions for other stocks or time periods.

## Data
The dataset used for this project contains:
- **Historical Data**:
  - Daily stock prices: Open, Close, High, Low
  - Volume of stocks traded
- **Time Range**: Last 6 months

### Data Preprocessing
- Normalize "Close" prices using MinMaxScaler for LSTM to ensure all values lie within a 0-1 range.
- Convert date columns to "datetime" format for compatibility with Prophet and Plotly visualizations.
- Prepare 60-day sliding window sequences for LSTM input.

## Methodology
### Step 1: Data Understanding
- Load historical stock data using the Yahoo Finance API.
- Conduct Exploratory Data Analysis to visualize and summarize key trends in the data.

### Step 2: Prophet Model
- Format the "Date" and "Close" columns for compatibility with Prophet's "ds" and "y" requirements.
- Train the model to identify long-term trends and seasonality in Amazon stock prices.
- Forecast for a 365-day period and visualize predictions with confidence intervals.

### Step 3: LSTM Model
- Normalize the data for improved performance using MinMaxScaler.
- Create sequences of 60 days as features with the next day’s price as the target.
- Train an LSTM model with two stacked LSTM layers and a Dense output layer for prediction.
- Evaluate performance using metrics like Mean Squared Error (MSE).

### Step 4: Model Comparison
- Compare Prophet and LSTM predictions for specific dates to understand their strengths and weaknesses.
- Visualize predictions and observed prices for better clarity.

## Results
### Key Insights from EDA
- Stock prices demonstrate clear seasonal patterns.
- Closing prices are influenced by trading volumes during high volatility periods.

### Model Performance
- **Prophet**:
  - Provides reliable confidence intervals and performs well with long-term trends.
- **LSTM**:
  - Captures short-term price fluctuations effectively and offers better day-to-day accuracy.

### Predictions
For a specific date (2024-11-20):
- **Prophet Predicted Price**: $201.80
- **LSTM Predicted Price**: $198.27

## Future Work
- **Extend Dataset**: Include more historical data for improved model robustness.
- **Additional Features**: Incorporate market sentiment, macroeconomic indicators, and industry news.
- **Model Deployment**: Build a real-time forecasting application using Flask or FastAPI.
- **Advanced Models**: Experiment with modern architectures like Transformers for better predictions.


