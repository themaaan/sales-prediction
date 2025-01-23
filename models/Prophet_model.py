import numpy as np
import pandas as pd
from prophet import Prophet
import plotly
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def prepare_data_for_prophet(data):
    """
    Prepares the dataset for Facebook Prophet.
    - Renames the date column to 'ds'.
    - Renames the target variable column to 'y'.
    """
    data = data[['date', 'courier_partners_online']].copy()  # Select relevant columns
    data.rename(columns={'date': 'ds', 'courier_partners_online': 'y'}, inplace=True)
    return data


def forecast_courier_prophet(data, plot = False):
    """
    Implements Facebook Prophet for forecasting courier partners online.
    """
    prophet_data = prepare_data_for_prophet(data)

    model = Prophet()
    model.fit(prophet_data)
    future = model.make_future_dataframe(periods=14)  # Forecast 14 days ahead
    forecast = model.predict(future)

    if plot:
        # Plot the forecast
        model.plot(forecast)
        plt.show()
        model.plot_components(forecast)
        plt.show()  
    forecast = forecast.rename(columns={"ds": "date", "yhat": "prophet_pred"})
    forecast = forecast[["date", "prophet_pred"]]

    forecast['date'] = pd.to_datetime(forecast['date'], errors='coerce')    
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  

    metric_df = forecast.set_index('date')[['prophet_pred']].join(data.set_index('date')[['courier_partners_online']]).reset_index()
    metric_df = data.merge(forecast, on='date', how='inner')
    score = r2_score(metric_df.courier_partners_online, metric_df.prophet_pred)
    
    return forecast.tail(14).reset_index()[["date", "prophet_pred"]], score