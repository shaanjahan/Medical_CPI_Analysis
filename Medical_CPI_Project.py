import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from fredapi import Fred
import statsmodels.api as sm
from sklearn.cluster import KMeans

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

def cpi_correlation_matrix(df):
    df_pct = df.pct_change().dropna() * 100  # % change for all series
    correlation = df_pct.corr()
    
    print("📊 Correlation Matrix (% Changes):\n")
    print(correlation)
    
    # Optional: Heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Between CPI Components')
    plt.tight_layout()
    plt.show()
    
def lagged_cross_correlation(df, lags=6):
    df_pct = df.pct_change().dropna() * 100
    base = df_pct['Medical CPI']
    results = {}

    for var in ['Food CPI', 'Core CPI']:
        corr_lags = [base.corr(df_pct[var].shift(lag)) for lag in range(lags+1)]
        results[var] = corr_lags

    pd.DataFrame(results, index=[f'Lag {i}' for i in range(lags+1)]).plot.bar(figsize=(10, 5), title="Lagged Correlations with Medical CPI")
    plt.axhline(0, color='gray', linestyle=':')
    plt.grid(True)
    plt.show()

cpi_correlation_matrix(df)
lagged_cross_correlation(df)

def classify_med_cpi_trends(df, window=6, n_clusters=3):
    df_pct = df[['Medical CPI']].pct_change().dropna() * 100

    # Rolling window features
    df_pct['Rolling Mean'] = df_pct['Medical CPI'].rolling(window).mean()
    df_pct['Rolling Volatility'] = df_pct['Medical CPI'].rolling(window).std()
    df_pct = df_pct.dropna()

    # Feature matrix for clustering
    features = df_pct[['Rolling Mean', 'Rolling Volatility']].copy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_pct['Trend Cluster'] = kmeans.fit_predict(features)

    # Visualization
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 6))
    for cluster in range(n_clusters):
        cluster_data = df_pct[df_pct['Trend Cluster'] == cluster]
        plt.plot(cluster_data.index, cluster_data['Medical CPI'], label=f'Regime {cluster}', lw=1.8)

    plt.title(f"Medical CPI Trend Classification ({n_clusters} Regimes)")
    plt.xlabel("Date")
    plt.ylabel("Medical CPI (% Change)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_pct

def summarize_trend_clusters(classified_df):
    summary = classified_df.groupby('Trend Cluster')[['Rolling Mean', 'Rolling Volatility']].mean()
    print("\n Regime Summary:\n")
    print(summary.round(2))

classified_df = classify_med_cpi_trends(df, window=6, n_clusters=3)
summarize_trend_clusters(classified_df)

def simulate_policy_scenarios(rf_model, scaler, baseline_input, food_shocks, core_shocks):
    food_base, core_base, med_last = baseline_input
    results = []

    for food_change in food_shocks:
        for core_change in core_shocks:
            food_input = food_base * (1 + food_change / 100)
            core_input = core_base * (1 + core_change / 100)

            X = [[food_input, core_input, med_last]]
            X_scaled = scaler.transform(X)
            med_pred = rf_model.predict(X_scaled)[0]

            results.append({
                'Food CPI Shock (%)': food_change,
                'Core CPI Shock (%)': core_change,
                'Predicted Medical CPI (% Change)': round(med_pred, 3)
            })
    return pd.DataFrame(results)

# Last known values (baseline inputs)
food_last = df_prep['Food CPI %'].iloc[-1]
core_last = df_prep['Core CPI %'].iloc[-1]
med_last = df_prep['Medical CPI'].iloc[-1]

# Policy shock scenarios
food_shocks = [0, 2, 5, 10]   # e.g. simulate 0%, +2%, +5%, +10% food CPI
core_shocks = [0, 2, 4]       # e.g. 0%, +2%, +4% core CPI

# Run the simulation
scenario_df = simulate_policy_scenarios(
    rf_model=rf,
    scaler=scaler,
    baseline_input=[food_last, core_last, med_last],
    food_shocks=food_shocks,
    core_shocks=core_shocks
)

print("\n Policy Simulation Results:\n")
print(scenario_df)
