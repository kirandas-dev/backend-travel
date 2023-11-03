from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

daily_revenue = pd.read_csv('https://raw.githubusercontent.com/kirandas-dev/data-ML/main/combined_time_series.csv', low_memory=False)

# Convert the 'date' column in your DataFrame to Timestamp objects
daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])

train_size = len(daily_revenue) - 400  # Leave the last 400 days as that belong to test set. 
train_data = daily_revenue[:train_size]
test_data = daily_revenue[train_size:]


class SARIMAXTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, order, seasonal_order, trend):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fit SARIMA model and make predictions
        model = SARIMAX(X, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=7)  # Forecast next 7 days

        return forecast

# Use the SARIMA parameters from your grid search
best_sarima_params = [(2, 1, 2), (0, 0, 0, 0), 't']  # Replace with your best parameters

# Create and train the SARIMA pipeline
sarima_pipeline = Pipeline([
    ('sarimax', SARIMAXTransformer(order=best_sarima_params[0], seasonal_order=best_sarima_params[1], trend=best_sarima_params[2]))
])

sarima_pipeline.fit(train_data['sales'])

model_filename= "/Models/forecasting/sarima_pipeline.joblib"

joblib.dump(sarima_pipeline, model_filename)


