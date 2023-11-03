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


@app.get('/getairfare/items/', status_code=200)
#def airfare_predict(startingAirport: str, destinationAirport: str, store_id: str):
def airfare_predict():
                # Make predictions based on user inputs
        input_data = pd.DataFrame({
            'startingAirport': ['ATL'],
            'destinationAirport': ['BOS'],
            'DayOfWeek': [5],  # Example: 5 for Friday
            'Month': [4],  # Example: 4 for April
            'Year': [2022],
            'DepartureHour': [8],  # Example: 8 for 8:00 AM
            'DepartureMinute': [0],  # Example: 0 for 0 minutes past the hour
        })
        input_data['startingAirport'] = loaded_encoders.transform(input_data['startingAirport'])
        input_data['destinationAirport'] = loaded_encoders.transform(input_data['destinationAirport'])


        # Make predictions using the loaded model
        predicte_airfare = model_load['model'].predict(input_data)
        predicte_airfare = predicte_airfare.tolist()
       

        # Return the predicted sales as a JSON response
        return JSONResponse(content={"Predicted Airfare": predicte_airfare})
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
