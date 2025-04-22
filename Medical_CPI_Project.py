import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from fredapi import Fred

def prepare_features(df):
    df = df.copy()
    df['Food CPI %'] = df['Food CPI'].pct_change() * 100
    df['Core CPI %'] = df['Core CPI'].pct_change() * 100
    df = df[['Food CPI %', 'Core CPI %', 'Medical CPI']].dropna()
    df['Food CPI Lag1'] = df['Food CPI %'].shift(1)
    df['Core CPI Lag1'] = df['Core CPI %'].shift(1)
    df['Med CPI Lag1'] = df['Medical CPI'].shift(1)
    return df.dropna()

def forecast_medical_cpi(df, n_months=6):
    df_prep = prepare_features(df)

    X = df_prep[['Food CPI Lag1', 'Core CPI Lag1', 'Med CPI Lag1']]
    y = df_prep['Medical CPI']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    last_date = df_prep.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='MS')
    food_last, core_last, med_last = df_prep['Food CPI %'].iloc[-1], df_prep['Core CPI %'].iloc[-1], df_prep['Medical CPI'].iloc[-1]

    forecasts = []
    for _ in range(n_months):
        input_scaled = scaler.transform([[food_last, core_last, med_last]])
        med_last = rf.predict(input_scaled)[0]
        forecasts.append(med_last)

    # Return model + forecast data
    return rf, scaler, df_prep, forecast_dates, forecasts

# Create & print forecast table

def print_forecast_table(df, forecast_dates, forecasts):
    hist_df = df[['Medical CPI']].iloc[-6:].copy().reset_index()
    hist_df.columns = ['Date', 'Medical CPI (% Change)']
    hist_df['Type'] = 'Actual'

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Medical CPI (% Change)': forecasts,
        'Type': 'Forecast'
    })

    combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    combined_df['Direction'] = combined_df['Medical CPI (% Change)'].diff().apply(
        lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change')
    )

    print("\n📈 Medical CPI (% Change): Last 6 Months + Next 6 Forecasts\n")
    print(combined_df.to_string(index=False))

    return combined_df

# Plotting forecast 

def plot_forecast_table(combined_df):
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 6))

    actual = combined_df[combined_df['Type'] == 'Actual']
    forecast = combined_df[combined_df['Type'] == 'Forecast']

    plt.plot(actual['Date'], actual['Medical CPI (% Change)'], label='Actual (Past 6 Months)', marker='o', lw=2)
    plt.plot(forecast['Date'], forecast['Medical CPI (% Change)'], label='Forecast (Next 6 Months)', linestyle='--', marker='o', lw=2)

    plt.xlim(actual['Date'].iloc[0], forecast['Date'].iloc[-1] + pd.DateOffset(months=1))
    plt.axhline(0, color='gray', linestyle=':')
    plt.title("Medical CPI (% Change): Actual vs Forecasted")
    plt.ylabel("Monthly % Change")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


fred = Fred(api_key='cb13b116bf78b290ea691bbf95c189bf')

# Fetch data
food_cpi = fred.get_series('CUSR0000SAF11')
overall_cpi = fred.get_series('CPIAUCSL')
core_cpi = fred.get_series('CPILFESL')
med_cpi = fred.get_series('MEDCPIM158SFRBCLE')
df = pd.concat([food_cpi, overall_cpi, core_cpi, med_cpi], axis=1)
df.columns = ['Food CPI', 'Overall CPI', 'Core CPI', 'Medical CPI']
df = df.dropna()

df.index = pd.to_datetime(df.index)
rf, scaler, df_prep, forecast_dates, forecasts = forecast_medical_cpi(df)
combined_df = print_forecast_table(df_prep, forecast_dates, forecasts)
plot_forecast_table(combined_df)
