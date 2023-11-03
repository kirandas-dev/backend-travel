import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


# Encode 'startingAirport' and 'destinationAirport' using label encoding
airport_label_encoder = LabelEncoder()

sampled_data= pd.read_csv('https://raw.githubusercontent.com/kirandas-dev/data-ML/main/sampled.csv', low_memory=False)


sampled_data['startingAirport'] = airport_label_encoder.fit_transform(sampled_data['startingAirport'])
sampled_data['destinationAirport'] = airport_label_encoder.transform(sampled_data['destinationAirport'])


# Data Preprocessing
sampled_data['flightDate'] = pd.to_datetime(sampled_data['flightDate'])
sampled_data['DayOfWeek'] = sampled_data['flightDate'].dt.dayofweek
sampled_data['Month'] = sampled_data['flightDate'].dt.month
sampled_data['Year'] = sampled_data['flightDate'].dt.year
sampled_data['DepartureTime'] = pd.to_datetime(sampled_data['segmentsDepartureTimeRaw'].str.split('||').str[0])
sampled_data['DepartureHour'] = sampled_data['DepartureTime'].dt.hour
sampled_data['DepartureMinute'] = sampled_data['DepartureTime'].dt.minute
sampled_data = pd.get_dummies(sampled_data, columns=['segmentsCabinCode'], prefix='Cabin')

# Feature Selection
features = ['startingAirport', 'destinationAirport', 'DayOfWeek', 'Month', 'Year', 'DepartureHour', 'DepartureMinute'] + [col for col in sampled_data.columns if col.startswith('Cabin_')]

# Extract features and target variable
X = sampled_data[['startingAirport', 'destinationAirport', 'DayOfWeek', 'Month', 'Year', 'DepartureHour', 'DepartureMinute']]
y = sampled_data['totalFare']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Define LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the LightGBM model
model = lgb.train(params, train_data, num_boost_round=100)

# Make predictions based on user inputs
user_input = pd.DataFrame({
    'startingAirport': ['ATL'],
    'destinationAirport': ['BOS'],
    'DayOfWeek': [5],  # Example: 5 for Friday
    'Month': [4],  # Example: 4 for April
    'Year': [2022],
    'DepartureHour': [8],  # Example: 8 for 8:00 AM
    'DepartureMinute': [0],  # Example: 0 for 0 minutes past the hour
})
user_input['startingAirport'] = airport_label_encoder.transform(user_input['startingAirport'])
user_input['destinationAirport'] = airport_label_encoder.transform(user_input['destinationAirport'])

# Predict the fare
predicted_fare = model.predict(user_input, num_iteration=model.best_iteration)

print("Predicted Fare: $", round(predicted_fare[0], 2))

model_and_encoders = {
        'model': model,
        'encoders': airport_label_encoder
    }
model_filename= "Models/predictive/lightgbm.joblib"

joblib.dump(model_and_encoders, model_filename)

