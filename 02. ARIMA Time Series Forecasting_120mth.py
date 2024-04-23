#!/usr/bin/env python
# coding: utf-8

# # ARIMA Time Series Forecasting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import itertools
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# ## Step 1: Load and Preprocess the Dataset

# Load the dataset
Canadian_data = pd.read_csv('/Users/kellyliu/Documents/GitHub/ARIMA-Time-Serie-Forecasting/Canadian Sales.csv')
Canadian_data.reset_index(inplace=True)
Canadian_data = Canadian_data[['REF_DATE', 'VALUE']]
Canadian_data['REF_DATE'] = pd.to_datetime(Canadian_data['REF_DATE'])

# ## Step 2: Check for Stationarity and Make the Data Stationary

def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return dftest[1]  # return p-value

# Test stationarity
p_value = test_stationarity(Canadian_data['VALUE'])
if p_value > 0.05:
    # First differencing
    Canadian_data['VALUE'] = Canadian_data['VALUE'].diff().dropna()
    print("Data differenced once.")
    # Plot data after differencing
    plt.figure(figsize=(10, 4))
    plt.plot(Canadian_data['REF_DATE'], Canadian_data['VALUE'], label='Differenced Data')
    plt.title('Data After Differencing')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Test stationarity again
    p_value = test_stationarity(Canadian_data['VALUE'].dropna())
    if p_value > 0.05:
        # Apply transformation, e.g., logarithmic
        Canadian_data['VALUE'] = np.log(Canadian_data['VALUE'].dropna())
        print("Log transformation applied.")
        # Plot data after log transformation
        plt.figure(figsize=(10, 4))
        plt.plot(Canadian_data['REF_DATE'], Canadian_data['VALUE'], label='Log Transformed Data')
        plt.title('Data After Log Transformation')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

# ## Step 3: Find the Best PDQ and Seasonal PDQ Parameters

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
aic_list = []

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(Canadian_data['VALUE'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            aic_list.append({'params': param, 'seasonal_params': param_seasonal, 'aic': results.aic})
        except:
            continue

best_model = min(aic_list, key=lambda x: x['aic'])
print("Best model: ", best_model)

# ## Step 4: Time-Series Cross-Validation

tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []
mae_scores = []
rmse_scores = []

for train_index, test_index in tscv.split(Canadian_data):
    train, test = Canadian_data.iloc[train_index], Canadian_data.iloc[test_index]
    model = sm.tsa.statespace.SARIMAX(train['VALUE'],
                                      order=best_model['params'],
                                      seasonal_order=best_model['seasonal_params'],
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    results = model.fit()
    predictions = results.forecast(steps=len(test))
    mse = mean_squared_error(test['VALUE'], predictions)
    mae = mean_absolute_error(test['VALUE'], predictions)
    rmse = np.sqrt(mse)
    mse_scores.append(mse)
    mae_scores.append(mae)
    rmse_scores.append(rmse)

    # Visual comparison of Forecast vs Actuals
    plt.figure(figsize=(12, 6))
    plt.plot(train['REF_DATE'], train['VALUE'], label='Training Data')
    plt.plot(test['REF_DATE'], test['VALUE'], label='Actual Test Data')
    plt.plot(test['REF_DATE'], predictions, label='Forecasted Data', color='red', linestyle='--')
    plt.title('Forecast vs Actuals')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

print(f'Mean MSE: {np.mean(mse_scores)}, Mean MAE: {np.mean(mae_scores)}, Mean RMSE: {np.mean(rmse_scores)}')