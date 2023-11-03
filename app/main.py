from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

model_filename= "/Models/predictive/lightgbm.joblib"

model_load = load(model_filename)
loaded_encoders = model_load['encoders']

# Check if the loading was successful
if loaded_encoders:
    print("Encoders loaded successfully.")
else:
    print("Encoders loading failed.")

@app.get("/")
def read_root():
    return {
        "Project Objectives": "This project deploys two different models as APIs:"
        
    }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Hello there. Welcome! Get some useful insights using our predictives analytics on the American Retail Store!'


@app.get("/predict_airfare")
def predict_airfare(startingAirport: str, destinationAirport: str, DayOfWeek: int, Month: int, Year: int, DepartureHour: int, DepartureMinute: int):
    # Create a user input DataFrame
    input_data = pd.DataFrame({
        'startingAirport': [startingAirport],
        'destinationAirport': [destinationAirport],
        'DayOfWeek': [DayOfWeek],
        'Month': [Month],
        'Year': [Year],
        'DepartureHour': [DepartureHour],
        'DepartureMinute': [DepartureMinute]
    })

    # Transform categorical features using the loaded encoders
    input_data['startingAirport'] = loaded_encoders.transform(input_data['startingAirport'])
    input_data['destinationAirport'] = loaded_encoders.transform(input_data['destinationAirport'])

    # Make predictions using the loaded model
    predicted_airfare = model_load['model'].predict(input_data)
    predicted_airfare = predicted_airfare.tolist()

    # Return the predicted airfare as a JSON response
    return JSONResponse(content={"Predicted Airfare": predicted_airfare})
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
