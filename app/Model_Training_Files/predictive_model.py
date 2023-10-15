# Import necessary libraries
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import joblib

import pandas as pd
import numpy as np
data = pd.read_csv('../data/processed/final_merged_events.csv', low_memory=False)


# Assuming 'date' is in a string format, convert it to datetime
data['date'] = pd.to_datetime(data['date'])

# Now you can use .dt accessor to extract date components
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year


import os
import pandas as pd
import lightgbm as lgb  # Import LightGBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from category_encoders import TargetEncoder
import joblib
import numpy as np

# List of store names
store_names = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']

# Define the groups based on your criteria
group1 = ['CA_1', 'CA_2', 'CA_3', 'CA_4']
group2 = ['TX_1', 'TX_2', 'TX_3']
group3 = ['WI_1', 'WI_2', 'WI_3']

# Initialize a dictionary to store the trained models and encoders for each group
models = {}

# Initialize lists to store RMSE for each group
rmse_scores = []

# Iterate over each group and filter the data accordingly
for group_idx, group in enumerate([group1, group2, group3]):
    # Filter the data for the current group
    group_data = data[data['store_id'].isin(group)]

    # Define features and target variable
    date_features = ['day_of_week', 'month', 'year']
    categorical_features = ['store_id', 'item_id']
    target = 'sales'

    # Split the data into training, validation, and test sets
    train_data, test_data = train_test_split(group_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Extract features and target variables for training, validation, and test sets
    X_train, y_train = train_data[date_features + categorical_features], train_data[target]
    X_val, y_val = val_data[date_features + categorical_features], val_data[target]
    X_test, y_test = test_data[date_features + categorical_features], test_data[target]

    # Initialize target encoders for categorical features
    target_encoders = {}
    for feature in categorical_features:
        target_encoder = TargetEncoder()
        X_train[feature] = target_encoder.fit_transform(X_train[feature], y_train)
        X_val[feature] = target_encoder.transform(X_val[feature])
        X_test[feature] = target_encoder.transform(X_test[feature])
        target_encoders[feature] = target_encoder

    # Initialize and train the LightGBM model on the encoded feature set
    model = lgb.LGBMRegressor()  # Use LightGBMRegressor
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    val_predictions = model.predict(X_val)

    # Evaluate the model on the validation set using Mean Absolute Error (MAE)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print(f'Group {group_idx + 1} - Validation MAE with Target Encoding: {val_mae}')

    # Store the trained model and encoders in the dictionary for this group
    model_and_encoders = {
        'model': model,
        'encoders': target_encoders
    }

    # Define the models directory
    models_dir = "../Model_Serving/models/predictive"  # Update the directory name
    os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the model file path for saving
    model_file_path = os.path.join(models_dir, f'model_group_{group_idx + 1}.joblib')

    # Save the model to the specified file path using joblib
    joblib.dump(model_and_encoders, model_file_path)

    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Calculate RMSE on the test set
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f'Group {group_idx + 1} - Test RMSE with Target Encoding: {rmse}')

    # Append the RMSE score to the list
    rmse_scores.append(rmse)

# Calculate the average RMSE across groups
average_rmse = np.mean(rmse_scores)
print(f'Average RMSE across groups: {average_rmse}')
