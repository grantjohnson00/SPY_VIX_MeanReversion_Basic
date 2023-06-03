import yfinance as yf
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Avg of 90 day T-bills over period
ann_rf = 0.0227

#METHODS
#`````````````````````````````````````````````````````````````````````````````````````````````````````
# Method for randomly simulated 1-year SPY Sharpe Ratios
def random_dates(start, end, n=10000):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end) - pd.DateOffset(years=1)  # Adjust the end date to ensure we have enough data for a 1-year period

    date_range = (end_date - start_date).days
    random_days = sorted(random.sample(range(date_range), n))

    return [start_date + pd.DateOffset(days=days) for days in random_days]

# Method for sharpe ratio calculation
def sharpe_ratio(returns, ann_rf):
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = (ann_ret - ann_rf) / ann_vol
    return sharpe
#`````````````````````````````````````````````````````````````````````````````````````````````````````

# Download VIX data
vixData = yf.download('^VIX', start="1993-01-29", end="2023-03-16")
vixData['PointsChange'] = vixData['Adj Close'] - vixData['Adj Close'].shift(1)

# Download SPY data
spyData = yf.download('SPY', start="1993-01-29", end="2023-03-16")

# Calculate daily returns
spyData['Daily_Return'] = spyData['Close'].pct_change()

# Merge the data
combData = pd.merge(vixData, spyData, on='Date', suffixes=('_Vix', '_Spy'))

# Calculate PointsChange statistics
pointsChange_std = vixData['PointsChange'].std()
pointsChange_avg = vixData['PointsChange'].mean()

# Calculate adjusted close statistics
adjClose_std = vixData['Adj Close'].std()
adjClose_mean = vixData['Adj Close'].mean()

# Empty DataFrame for filtered data (get same columns)
vixSignals = pd.DataFrame(columns=combData.columns)

# Loop through DataFrame to find days that meet desired criteria
for index, row in combData.iterrows():
    if row['Adj Close_Vix'] >= adjClose_mean + 1 * adjClose_std and row['PointsChange'] >= 4 * pointsChange_std:
        vixSignals = pd.concat([vixSignals, pd.DataFrame(row).transpose()])
        
# Empty DataFrame for trade signals
tradeSignals = pd.DataFrame(columns=['Buy_Date', 'Sell_Date', 'Buy_Price', 'Sell_Price', 'Return_Percent', 'Sharpe_Ratio'])

for index, row in vixSignals.iterrows():
    buyDate = index
    # Buy-in (100%) two weeks later
    addDate = buyDate + pd.DateOffset(weeks=2)
    if addDate in spyData.index:
        buyPrice = spyData.loc[addDate]['Close']

    sellDate = addDate + pd.DateOffset(years=1)
    if sellDate not in spyData.index:
    # Find the next available date after the desired sell date
        sellDate = spyData.index[spyData.index.get_indexer([sellDate], method='nearest')[0]]

    sellPrice = spyData.loc[sellDate]['Close']
    returns = spyData.loc[addDate:sellDate, 'Daily_Return'].dropna()
    sharpe = sharpe_ratio(returns, ann_rf)

    new_row = pd.DataFrame({'Buy_Date': [buyDate], 'Sell_Date': [sellDate], 
                            'Buy_Price': [buyPrice], 'Sell_Price': [sellPrice], 
                            'Return_Percent': [(sellPrice - buyPrice) / buyPrice], 
                            'Sharpe_Ratio': [sharpe]})

    tradeSignals = pd.concat([tradeSignals, new_row]).reset_index(drop=True)

average_Vix_TimingSignal_Sharpe = np.mean(tradeSignals["Sharpe_Ratio"])
average_VIX_TimingSignal_Return = np.mean(tradeSignals["Return_Percent"])*100
median_VIX_TimingSignal_Return = np.median(tradeSignals["Return_Percent"]*100)


#RANDOM 1-YEAR SHARPE SIMULATION

# Generate 35 random start dates within the SPY data
random_start_dates = random_dates("1993-01-29", "2023-03-16")

# Calculate the Sharpe Ratio for each 1-year period starting from the random start dates
random_sharpe_ratios = []
for start_date in random_start_dates:
    end_date = start_date + pd.DateOffset(years=1)
    if end_date not in spyData.index:
        end_date = spyData.index[spyData.index.get_indexer([end_date], method='nearest')[0]]

    returns = spyData.loc[start_date:end_date, 'Daily_Return'].dropna()
    random_sharpe_ratios.append(sharpe_ratio(returns, ann_rf))

# Calculate the average Sharpe Ratio across all the 35 periods
average_random_sharpe = np.mean(random_sharpe_ratios)


# Calculate the standard deviation of the random 1-year Sharpe Ratios
std_random_sharpe = np.std(random_sharpe_ratios)

# Calculate the 99% confidence interval
confidence_level = 0.99
degrees_of_freedom = len(random_sharpe_ratios) - 1
mean_random_sharpe = np.mean(random_sharpe_ratios)
confidence_interval = stats.t.interval(confidence_level, degrees_of_freedom, loc=mean_random_sharpe, scale=std_random_sharpe / np.sqrt(degrees_of_freedom))
print("Standard deviation of random 1-year Sharpe Ratios:", std_random_sharpe)
print(f"95% confidence interval for random 1-year Sharpe Ratios: {confidence_interval}")




#REGRESSION STATS

# Calculate daily returns for VIX
vixData['Daily_Return'] = vixData['Adj Close'].pct_change()
# Merge data
combData2 = pd.merge(vixData, spyData, on='Date', suffixes=('_Vix', '_Spy'))
# Drop rows with missing data (due to pct_change)
combData2.dropna(inplace=True)
# Set the independent (X) and dependent (y) variables
X = combData2['Daily_Return_Spy']
y = combData2['Daily_Return_Vix']
# Add a constant to the independent variable (X) for the regression model
X = sm.add_constant(X)
# Perform the linear regression
model = sm.OLS(y, X).fit()
# Print regression results
print(model.summary())


#REGRESSION PLOT

# Scatter plot of daily VIX returns against daily SPY returns
plt.scatter(combData2['Daily_Return_Spy'], combData2['Daily_Return_Vix'], alpha=0.5)
plt.xlabel('SPY Daily Returns')
plt.ylabel('VIX Daily Returns')
plt.title('Scatter Plot of Daily VIX Returns vs Daily SPY Returns')

# Regression line
regression_line_x = np.linspace(combData2['Daily_Return_Spy'].min(), combData2['Daily_Return_Spy'].max(), 100)
regression_line_y = model.params[0] + model.params[1] * regression_line_x
plt.plot(regression_line_x, regression_line_y, color='red')

# Custom legend
legend_line = Line2D([0], [0], color='red', lw=2, label='Regression Line')
plt.legend(handles=[legend_line])

# Show the plot
plt.show()


#BENCHMARK

#Benchmark SPY Sharpe over entire period(2.27% avg 3-M T Bill rate)
entirePeriodSpySharpe = sharpe_ratio(spyData['Adj Close'].pct_change(), ann_rf)
print("The sharpe ratio of SPY from 01-29-1993 to 03-16-2023 was:", entirePeriodSpySharpe)

