# Import the necessary Python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm

# Loading the sales data and pre-processing
data=pd.read_csv('New_motor_vehicle_sales_data.csv', parse_dates = True, index_col = 'REF_DATE')
data = data.sort_index()
# Display the shape of the data
data.shape
# Display the first few rows of the data
data.head()
# check for missing values
data.isnull().sum()
# Check for duplicates
data.duplicated().sum()
# Display the unique value within each columns
data.nunique()

# Display the  unique values in the 'GEO' column
data['GEO'].unique()
data.columns
# Split the data with 'GEO="Canada"'
df = data[data['GEO'] == 'Canada']
df = df[df['Vehicle type'] == 'Total, new motor vehicles']
df = df[df['Origin of manufacture'] == 'Total, country of manufacture']
df = df[df['Sales'] == 'Dollars']
df=df['VALUE']
df = df.loc['2010-01-01 00:00:00':'2019-12-01 00:00:00']

# Adjust figure size
plt.figure(figsize=(20, 10))

# Plot the lineplot
sns.lineplot(data=df)#, x='REF_DATE', y='VALUE')#, hue='GEO')
# Add more space for y-axis labels
#plt.ylim(bottom=0)
# Set labels and title
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Sales Value vs Year for Canada')
# Show the plot
plt.show()


# ARIMA data analysis and model fitting
decomposition = sm.tsa.seasonal_decompose(df, model = 'additive')
fig = decomposition.plot()
plt.show()

from statsmodels.tsa.stattools import adfuller
adftest = adfuller(df)

print('pvalue of adfuller test is: ', adftest[1])

split = int(len(df)*0.8)
train = df[:split]
test = df[split:]

start = len(train)
end = len(train) + len(test)

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(13,1,6)).fit()
model.summary()
import warnings
warnings.filterwarnings('ignore')

pred = model.predict(start=len(train), end=len(df)-1)
pred.index = df.index[start:end]
print(pred)


# Calculation of Error between predicted and test data
from sklearn.metrics import mean_squared_error
error = np.sqrt(mean_squared_error(test, pred))
print(f"\nThe standard deviation of prediction:test is {error}")
print(f"The mean of test data is {test.mean()}")

plt.figure(figsize=(20, 10))

# Plot the lineplot
sns.lineplot(data=train)
sns.lineplot(data=test)
sns.lineplot(data=pred)
plt.show()


# ACF, PACF plots for getting p,d, and q
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig,axes = plt.subplots(2,2)
#axes[0,0].plot(df)
plot_acf(df, ax = axes[0,0])
plot_acf(df.diff().dropna(), ax= axes[0,1])
#axes[1,0].plot(df)
plot_pacf(df.diff().dropna(), ax = axes[1,0])
plot_pacf(df.dropna(), ax = axes[1,1])
plt.show()

# Use of iteration to get p,d, and q
import itertools
p = range(0,13)
q = range(0,13)
d = range(0,2)

pdq_combination = list(itertools.product(p,d,q))

rmse = []
order1 = []

for pdq in pdq_combination:
    try:
        model = ARIMA(train, order= pdq).fit()
        pred = model.predict(start = len(train), end = (len(df)-1))
        error = np.sqrt(mean_squared_error(test, pred))
        order1.append(pdq)
        rmse.append(error)
    except:
        continue

results = pd.DataFrame(index = order1, data = rmse, columns= ['RMSE'])
results.to_csv('ARIMA_result.csv')


# Forecasting with fitted ARIMA model next 12 months sales data
model = ARIMA(df, order=(13,1,6)).fit()
model.summary()

forecast = model.predict(start=len(df)-1, end=len(df)+12)
print(forecast)

plt.figure(figsize=(20, 10))

# Plot the lineplot
sns.lineplot(data=df)
sns.lineplot(data=forecast)
plt.show()
