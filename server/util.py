import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    loc_index = __data_columns.index(location.lower())
    
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations

def load_data_and_model():
    global __data_columns
    global __locations
    global __model
    
    with open('c:/Users/yash/Desktop/Machine Learning/House Price Prediction/model/columns.json',) as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open('c:/Users/yash/Desktop/Machine Learning/House Price Prediction//model/bengaluru_home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)

if __name__ == "__main__":
    load_data_and_model()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))