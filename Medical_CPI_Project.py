import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns

# FRED API setup
fred = Fred(api_key='cb13b116bf78b290ea691bbf95c189bf')

# Fetch data
food_cpi = fred.get_series('CUSR0000SAF11')
overall_cpi = fred.get_series('CPIAUCSL')
core_cpi = fred.get_series('CPILFESL')
med_cpi = fred.get_series('MEDCPIM158SFRBCLE')

# Combine into a DataFrame
df = pd.concat([food_cpi, overall_cpi, core_cpi, med_cpi], axis=1)
df.columns = ['Food CPI', 'Overall CPI', 'Core CPI', 'Medical CPI']
df = df.dropna()

df.index = pd.to_datetime(df.index)

df_norm = df / df.iloc[0] * 100

plt.style.use('dark_background')
df_norm.plot(figsize=(14, 6), title='CPI Index (Normalized to 100)', lw=2)
plt.ylabel('Index (Base = 100)')
plt.grid(True)
plt.show()

df_pct_change = df.pct_change() * 100

volatility = df_pct_change.rolling(window=12).std()

df_pct_change.plot(figsize=(14, 6), title='Monthly % Change in CPI Categories', lw=1.5)
plt.ylabel('% Change')
plt.grid(True)
plt.show()

volatility.plot(figsize=(14, 6), title='Rolling 12-Month Volatility in CPI Categories', lw=2)
plt.ylabel('Volatility (% Std Dev)')
plt.grid(True)
plt.show()

pre_pandemic = df.loc[:'2020-02-01']
post_pandemic = df.loc['2020-03-01':]

growth_pre = (pre_pandemic.iloc[-1] / pre_pandemic.iloc[0] - 1) * 100
growth_post = (post_pandemic.iloc[-1] / post_pandemic.iloc[0] - 1) * 100

growth_df = pd.DataFrame({'Pre-Pandemic': growth_pre, 'Post-Pandemic': growth_post})
print(growth_df)

growth_df.plot(kind='bar', figsize=(10, 5), title='Cumulative CPI Growth Pre vs Post Pandemic')
plt.ylabel('Growth (%)')
plt.grid(True)
plt.show()

correlation = df_pct_change.corr()
print("Correlation Matrix:\n", correlation)


sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("CPI % Change Correlation Matrix")
plt.show()

def forecast_medical_cpi_with_direction(df, n_months=6):

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    df['Food CPI %'] = df['Food CPI'].pct_change() * 100
    df['Core CPI %'] = df['Core CPI'].pct_change() * 100

    df = df[['Food CPI %', 'Core CPI %', 'Medical CPI']].dropna()


    df['Food CPI Lag1'] = df['Food CPI %'].shift(1)
    df['Core CPI Lag1'] = df['Core CPI %'].shift(1)
    df['Med CPI Lag1'] = df['Medical CPI'].shift(1)
    df = df.dropna()

    X = df[['Food CPI Lag1', 'Core CPI Lag1', 'Med CPI Lag1']]
    y = df['Medical CPI']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)


    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='MS')

    food_last = df['Food CPI %'].iloc[-1]
    core_last = df['Core CPI %'].iloc[-1]
    med_last = df['Medical CPI'].iloc[-1]

    forecasts = []

    for _ in range(n_months):
        input_data = [[food_last, core_last, med_last]]
        input_scaled = scaler.transform(input_data)
        next_med = rf.predict(input_scaled)[0]
        forecasts.append(next_med)
        med_last = next_med  


    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Medical CPI (% Change)': forecasts,
        'Type': 'Forecast'
    })


    hist_df = df[['Medical CPI']].iloc[-6:].copy().reset_index()
    hist_df.columns = ['Date', 'Medical CPI (% Change)']
    hist_df['Type'] = 'Actual'


    full_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    full_df['Direction'] = full_df['Medical CPI (% Change)'].diff().apply(
        lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change')
    )


    print("\n📈 Medical CPI (% Change): Last 6 Months + Next 6 Forecasts\n")
    print(full_df.to_string(index=False))


    def plot_medical_cpi_forecast(df, n_months=2):

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    df['Food CPI %'] = df['Food CPI'].pct_change() * 100
    df['Core CPI %'] = df['Core CPI'].pct_change() * 100
    df = df[['Food CPI %', 'Core CPI %', 'Medical CPI']].dropna()
    df['Food CPI Lag1'] = df['Food CPI %'].shift(1)
    df['Core CPI Lag1'] = df['Core CPI %'].shift(1)
    df['Med CPI Lag1'] = df['Medical CPI'].shift(1)
    df = df.dropna()

    X = df[['Food CPI Lag1', 'Core CPI Lag1', 'Med CPI Lag1']]
    y = df['Medical CPI']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='MS')
    food_last = df['Food CPI %'].iloc[-1]
    core_last = df['Core CPI %'].iloc[-1]
    med_last = df['Medical CPI'].iloc[-1]

    forecasts = []
    for _ in range(n_months):
        input_data = [[food_last, core_last, med_last]]
        input_scaled = scaler.transform(input_data)
        next_med = rf.predict(input_scaled)[0]
        forecasts.append(next_med)
        med_last = next_med

    hist_df = df[['Medical CPI']].iloc[-6:].copy().reset_index()
    hist_df.columns = ['Date', 'Medical CPI (% Change)']
    hist_df['Type'] = 'Actual'

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Medical CPI (% Change)': forecasts,
        'Type': 'Forecast'
    })

    combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)

    plt.style.use('dark_background')
    plt.figure(figsize=(14, 6))

    actual = combined_df[combined_df['Type'] == 'Actual']
    forecast = combined_df[combined_df['Type'] == 'Forecast']

    plt.plot(actual['Date'], actual['Medical CPI (% Change)'], label='Actual (Past 6 Months)', marker='o', lw=2)
    plt.plot(forecast['Date'], forecast['Medical CPI (% Change)'], label='Forecast (Next 6 Months)', linestyle='--', marker='o', lw=2)

    last_plot_date = forecast['Date'].iloc[-1] + pd.DateOffset(months=1)
    plt.xlim(actual['Date'].iloc[0], last_plot_date)

    plt.axhline(0, color='gray', linestyle=':')
    plt.title("Medical CPI (% Change): Actual vs Forecasted")
    plt.ylabel("Monthly % Change")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_medical_cpi_forecast(df)

forecast_medical_cpi_with_direction(df)