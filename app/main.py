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


@app.get("/kiran_shop_predict_airfare_best_segment")
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

@app.get("/steff_shop_predict_airfare")
def predict_airfare(startingAirport: str, destinationAirport: str, FlightDay:int, FlightMonth:int, FlightYear:int, DepartureHour:int, DepartureMinute:int, DepartureSecond:int, isBasicEconomy:int, isRefundable:int, isNonStop:int, Days_Until_Flight:int, segmentsCabinCode:str):
    model_filename_segment = f"/Models/Steffi_Travel_Shop/Steffi_XGBoost_model.joblib"
    loaded_data_segment = load(model_filename_segment)
    Lab_enc = loaded_data_segment['label_encoder']
    xgb_model = loaded_data_segment['xgb_model']
    
    User_Request_XGBmodel3 = pd.DataFrame({
        'startingAirport': [startingAirport],
        'destinationAirport': [destinationAirport],
        'FlightDay': [FlightDay],
        'FlightMonth': [FlightMonth],
        'FlightYear': [FlightYear],
        'DepartureHour': [DepartureHour],
        'DepartureMinute': [DepartureMinute],
        'DepartureSecond': [DepartureSecond],
        'isBasicEconomy' : [isBasicEconomy], 
        'isRefundable':[isRefundable], 
        'isNonStop':[isNonStop], 
        'Days_Until_Flight':[Days_Until_Flight],
        'segmentsCabinCode': [segmentsCabinCode]
    })

    User_Request_XGBmodel3['startingAirport']=Lab_enc.fit_transform(User_Request_XGBmodel3['startingAirport'])
    User_Request_XGBmodel3['destinationAirport']=Lab_enc.fit_transform(User_Request_XGBmodel3['destinationAirport'])
    User_Request_XGBmodel3['segmentsCabinCode']=Lab_enc.fit_transform(User_Request_XGBmodel3['segmentsCabinCode'])
    Predicted_fare_XGBModel3 =  xgb_model.predict(User_Request_XGBmodel3)
    lowest_airfare_float = float(Predicted_fare_XGBModel3[0])
    return JSONResponse(content={ "Predicted Airfare": round(lowest_airfare_float, 2)})

@app.get("/Saumya_shop_predict_airfare")
def predict_airfare(
    startingAirport: str,
    destinationAirport: str,
    departure_year: int,
    departure_month: int,
    departure_day: int,
    departure_hour: int,
    departure_minute: int,
    cabin_code: str):
    # Create a DataFrame with the calculated values
    data = {
        'startingAirport': [startingAirport],
        'destinationAirport': [destinationAirport],
        'departure_year': [departure_year],
        'departure_month': [departure_month],
        'departure_day': [departure_day],
        'departure_hour': [departure_hour],
        'departure_minute': [departure_minute],
        'cabin_code': [cabin_code]
    }
    # Load the lgbm model and label encoder
    label_encoder = load('/Models/Saumya_Travel_Shop/label_encoder.joblib')
 
    lgb_model = load('/Models/Saumya_Travel_Shop/model_lgb.joblib')
    X_val = pd.DataFrame(data)

    airport_mapping = {
        'ATL': 904571,
        'BOS': 989175,
        'CLT': 888283,
        'DEN': 787482,
        'DFW': 929732,
        'DTW': 747751,
        'EWR': 713693,
        'IAD': 594215,
        'JFK': 692376,
        'LAX': 1352275,
        'LGA': 1032726,
        'MIA': 928766,
        'OAK': 527105,
        'ORD': 917732,
        'PHL': 785039,
        'SFO': 949046
    }

    # Apply airport_mapping to 'startingAirport' and 'destinationAirport'
    X_val['startingAirport'] = X_val['startingAirport'].replace(airport_mapping)
    X_val['destinationAirport'] = X_val['destinationAirport'].replace(airport_mapping)

    # Encoding 'cabin_code' using label_encoder
    X_val['cabin_code'] = label_encoder.transform(X_val['cabin_code'])

    # Implement your airfare prediction logic here
    # Use the loaded lgb_model to predict airfare
    pred = lgb_model.predict(X_val)

    lowest_airfare_float = float(pred[0])

    # Create a JSON response with the prediction
    
    return JSONResponse(content={ "Predicted Airfare": round(lowest_airfare_float, 2)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
