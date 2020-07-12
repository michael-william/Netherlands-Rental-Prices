# Data manipulation
import pandas as pd
import numpy as np
import streamlit as st
from geopy.geocoders import Nominatim

# ML libraries
import pycaret.regression as py
from pycaret.regression import *

# Visualizations
from matplotlib import pyplot as plt
import seaborn as sns
import chart_studio.plotly as pl
import plotly.graph_objs as go

st.write("""
# Simple Rental Prediction App
This app predicts the rental price of a property in the Netherlands!
All data is supplied via kamernet.nl with active properties listed for rent from July 2019 - March 2020. 
The app was developed using PyCaret with a Random Forrest model to predict the price of a property based on user inputs.
""")
@st.cache(allow_output_mutation=True)
def load_data():
    source = 'https://github.com/michael-william/Netherlands-Rental-Prices/raw/master/data/ml_data.csv'
    data=pd.read_csv(source, index_col=0)
    return data

with st.spinner(text="Loading data"):
    data = load_data()
st.success('Data laoded!')

@st.cache(allow_output_mutation=True)
def run_model():
    clf = py.setup(data, target = 'rent', silent=True)
    rf_model = py.create_model('rf', fold=5, verbose=False)
    model = py.finalize_model(rf_model)
    return model


def user_input_features():
        square_meters = st.sidebar.slider('Area in square meters', 6, 675, 56)
        locator = Nominatim(user_agent='myGeocoder')
        address = st.sidebar.text_input("Address of property", "Spaarndammerstraat 35 Amsterdam")
        location = locator.geocode(address)
        longitude = np.round(location.longitude,4)
        latitude = np.round(location.latitude,4)
        st.sidebar.text('City: '+(location.raw['display_name'].split(',')[3]))
        st.sidebar.text('Longitude: '+str(longitude))
        st.sidebar.text('Latitude: '+str(latitude))
        #latitude = st.sidebar.slider('Latitude', 50.770041, 53.333967, 51.2)
        #longitude = st.sidebar.slider('Longitude', 3.554188, 7.036756, 5.2)
        p_type = st.sidebar.selectbox('Type',['Apartment', 'Studio', 'Room', 'Anti-squat', 'Student residence'])
        shared = st.sidebar.selectbox('Shared',['No','Yes'])
        data = {'areaSqm': square_meters,
                'longitude': longitude,
                'latitude': latitude,
                'propertyType': p_type,
                'shared': shared}
        features = pd.DataFrame(data, columns = ['areaSqm','longitude','latitude', 'propertyType', 'shared'], index=[0])
        return features


with st.spinner("Creating Random Forrest model..this sould take about 30 seconds :-)"):
    model = run_model()
st.success('Model creation complete!')

def main():
    

    st.sidebar.header('Rental Parameters')
    
    user_df = user_input_features()
    #temp_df = temp_df

    st.subheader('Rental parameters')

    st.write(user_df)
    latitude = user_df.latitude[0]
    longitude = user_df.longitude[0]
    prop_t = user_df.propertyType[0]
    share = user_df.shared[0]

    def predict():
        prediction = py.predict_model(model,data=user_df)['Label']
        predict_per_sqm = prediction/user_df['areaSqm'][0]
        final_prediction = 'ML prediction: '+'€'+str(np.round(py.predict_model(model,data=user_df)['Label'][0],2))
        final_per_sqm = 'Price per sqm from model: '+ '€'+str(np.round(predict_per_sqm[0],2))   
        st.subheader('Monthly rental prediction')
        st.write(final_prediction)
        st.write(final_per_sqm)
        #st.write('ML model error range: +/- '+'€'+str(105)+' monthly rent')

    
    def temp_df():
        temp_df = data.copy()
        temp_df['long_from_origin'] = [abs(x-longitude) for x in temp_df['longitude']]
        temp_df['lat_from_origin'] = [abs(x-latitude) for x in temp_df['latitude']]
        temp_df['euro_per_sqm'] = np.ceil(temp_df.rent/temp_df.areaSqm)
        temp_result = temp_df[(temp_df['long_from_origin']<0.002) & (temp_df['lat_from_origin']<0.002) & (temp_df['propertyType'] == prop_t)]
        
        return temp_result

    if st.sidebar.button('Predict'):
        predict()
        temp = temp_df()
        st.subheader(str(len(temp))+' Other listings within 200 meters of your address')
        temp_df()
        st.write(temp[['areaSqm','rent','euro_per_sqm', 'propertyType', 'shared']])
        st.write('Average rent of other nearby listings: €'+ str(np.round(temp.rent.mean(),2)))
        st.write('Average euro per sqm of other nearby listings: €'+str(np.round(temp.euro_per_sqm.mean(),2)))
    st.subheader('')

if __name__ == "__main__":
    main()




