import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from geopy.geocoders import Nominatim

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score

st.write("""
# Simple Rental Prediction App
This app predicts the rental price of a property in the Netherlands!
All data is supplied via kamernet.nl. The app runs an XGBRegressor
model to predict the price of a property based on user inputs.
""")

@st.cache
def load_data(filename=None):
    data_source = 'https://github.com/michael-william/Netherlands-Rental-Prices/raw/master/final_ml_data.csv'
    df=pd.read_csv(data_source)
    y = df.rent
    X = df[['areaSqm','longitude','latitude','propertyType']]
    return df, X, y

df, X, y = load_data()

@st.cache(allow_output_mutation=True)
def run_ml_model():    
    categorical_cols = ['propertyType']

    numerical_cols = ['areaSqm','longitude','latitude']

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('label', OneHotEncoder(handle_unknown='ignore'))])

    # Grouping numeric and categorical preprocessing 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Define model
    model = XGBRegressor(n_estimators=1100, learning_rate=0.05)

    # Pipeline for processing the data and defining the model
    clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)])

    clf.fit(X,y)
    mae = '€'+str(111)
    return clf, model, mae

clf, model, mae = run_ml_model()

def user_input_features():
        square_meters = st.sidebar.slider('Area in square meters', 6, 675, 56)
        locator = Nominatim(user_agent='myGeocoder')
        address = st.sidebar.text_input("Address of property", "Spaarndammerstraat 35 1091GS")
        location = locator.geocode(address)
        longitude = round(location.longitude,4)
        latitude = round(location.latitude,4)
        st.sidebar.text('City: '+(location.raw['display_name'].split(',')[3]))
        st.sidebar.text('Longitude: '+str(longitude))
        st.sidebar.text('Latitude: '+str(latitude))
        #latitude = st.sidebar.slider('Latitude', 50.770041, 53.333967, 51.2)
        #longitude = st.sidebar.slider('Longitude', 3.554188, 7.036756, 5.2)
        p_type = st.sidebar.selectbox('Apartment',['Room', 'Studio', 'Apartment', 'Anti-squat', 'Student residence'])
        data = {'areaSqm': square_meters,
                'longitude': longitude,
                'latitude': latitude,
                'propertyType': p_type}
        features = pd.DataFrame(data, columns = ['areaSqm','longitude','latitude', 'propertyType'], index=[0])
        return features

def main():
    

    st.sidebar.header('Rental Parameters')


    user_df = user_input_features()

    st.subheader('Rental parameters')

    st.write(user_df)

    load_data()

    

    def predict():
        prediction = '€'+str(int(round(clf.predict(user_df)[0],0)))

        st.subheader('Monthly rental prediction')
        st.write(prediction)

    if st.sidebar.button('Predict'):
        predict()

        st.subheader('+/- range')
        st.write(mae)

if __name__ == "__main__":
    main()




