from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "Project Objectives": "This project deploys two different models as APIs:"
        
    }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Hello there. Welcome! Get some useful insights using our predictives analytics on the American Airlines!'


@app.get("/predict_airfare_best_segment")
def predict_airfare(startingAirport: str, destinationAirport: str, DayOfWeek: int, Month: int, Year: int, DepartureHour: int, DepartureMinute: int, CabinCode: str):

    lowest_airfare = float('inf')  # Initialize with positive infinity
    best_segment = None

    for segment_number in range(1, 5):  # Assuming you have models for segments 1, 2, 3, and 4
        try:
            # Load the distance predictor model
            model_filename_distance_predictor = f"/Models/Kiran_Travel_Shop/distanceforsegment{segment_number}.joblib"
            loaded_data_distance_predictor = load(model_filename_distance_predictor)
            distance_model = loaded_data_distance_predictor['xgb_model']

            # Create user input DataFrame for distance prediction
            user_input_distance = pd.DataFrame({
                'ArrTime': [destinationAirport],
                'DepAirport': [startingAirport],
            })
            user_input_distance['DepAirport'] = loaded_data_distance_predictor['airport_label_encoder'].transform(user_input_distance['DepAirport'])
            user_input_distance['ArrTime'] = loaded_data_distance_predictor['airport_label_encoder'].transform(user_input_distance['ArrTime'])
            predicted_distance = distance_model.predict(user_input_distance)

            # Load the duration predictor model
            model_filename_duration_predictor = f"/Models/Kiran_Travel_Shop/durationforsegment{segment_number}.joblib"
            loaded_data_duration_predictor = load(model_filename_duration_predictor)
            duration_model = loaded_data_duration_predictor['xgb_model']

            # Create user input DataFrame for duration prediction
            user_input_duration = pd.DataFrame({
                'ArrTime': [destinationAirport],
                'DepAirport': [startingAirport],
            })
            user_input_duration['DepAirport'] = loaded_data_duration_predictor['airport_label_encoder'].transform(user_input_duration['DepAirport'])
            user_input_duration['ArrTime'] = loaded_data_duration_predictor['airport_label_encoder'].transform(user_input_duration['ArrTime'])
            predicted_duration = duration_model.predict(user_input_duration)

            # Load the segment model
            model_filename_segment = f"/Models/Kiran_Travel_Shop/segment{segment_number}.joblib"
            loaded_data_segment = load(model_filename_segment)
            xgb_model = loaded_data_segment['xgb_model']

            # Create user input DataFrame for the current segment
            user_input = pd.DataFrame({
                'Duration': [predicted_duration[0]],
                'Distance': [predicted_distance[0]],
                'no_of_segment': [segment_number],
                'dep_day_of_week': [DayOfWeek],
                'dep_month': [Month],
                'arr_day_of_week': [DayOfWeek],
                'arr_month': [Month],
                'dep_hour': [DepartureHour],
                'dep_minute': [DepartureMinute],
                'arr_hour': [8],
                'arr_minute': [0],
                'ArrTime': [destinationAirport],
                'DepAirport': [startingAirport],
                'CabinCode': [CabinCode],
            })
            user_input['DepAirport'] = loaded_data_segment['airport_label_encoder'].transform(user_input['DepAirport'])
            user_input['ArrTime'] = loaded_data_segment['airport_label_encoder'].transform(user_input['ArrTime'])
            user_input['CabinCode'] = loaded_data_segment['cabin_code_encoder'].transform(user_input['CabinCode'])
            user_input['dep_day_of_week'] = loaded_data_segment['label_encoder'].transform(user_input['dep_day_of_week'])
            user_input['arr_day_of_week'] = loaded_data_segment['label_encoder'].transform(user_input['arr_day_of_week'])
            user_input['arr_hour'] = user_input['dep_hour'] + user_input['Duration'] // 60
            user_input['arr_minute'] = user_input['dep_minute'] + user_input['Duration'] % 60

            # Predict airfare for the current segment
            predicted_fare = xgb_model.predict(user_input)

            # Check if the current segment has the lowest airfare
            if predicted_fare[0] < lowest_airfare:
                lowest_airfare = predicted_fare[0]
                best_segment = segment_number

        except Exception as e:
            print(f"Error predicting for segment {segment_number}: {e}")

    print(f"Best Segment: {best_segment}, Predicted Fare: ${round(lowest_airfare, 2)}")

    # Convert lowest_airfare to a standard Python float before returning the JSON response
    lowest_airfare_float = float(lowest_airfare)

    # Return the predicted airfare of the best segment as a JSON response
    return JSONResponse(content={"Best Segment": best_segment, "Predicted Airfare": round(lowest_airfare_float, 2)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
