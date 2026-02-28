import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost
import joblib
import streamlit as st

df = pd.read_csv(r'houses_Madrid.csv')

df = df[['id' ,'subtitle', 'sq_mt_built', 'n_rooms', 'n_bathrooms', 
         'buy_price', 'buy_price_by_area', 'has_parking']]

df['n_bathrooms'] = df['n_bathrooms'].fillna(1)

df.drop(df.index[df.n_rooms == 0], axis= 0, inplace= True)

df['n_bathrooms'] = df['n_bathrooms'].astype(int)
         
df['id'] = df.index
df.drop(columns= 'id', inplace=True)

#df['subtitle'] = df['subtitle'].str.split(',', expand= True)[0]

#df = df.rename(columns= {'subtitle' : 'District'})

df['has_parking'] = df['has_parking'].astype(int)

df['subtitle'] = df['subtitle'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df['subtitle'] = df['subtitle'].str.replace(' ', '_').str.replace('-', '_')
df = df.rename(columns={'subtitle': 'District'})

df1 = pd.get_dummies(df, columns=['District'], drop_first=True)

df1.columns = df1.columns.str.replace(' ', '_').str.replace('-', '_')

x = df1.drop(columns=['buy_price', 'buy_price_by_area'])
y = df1['buy_price'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

XGR = xgboost.XGBRegressor() #modeling

XGR.fit(x_train, y_train) # training

y_pred = XGR.predict(x_test) #testing

y_pred.round(2)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print('RMSE:', np.sqrt(mse))
print('R2 Score:', r2_score(y_test,y_pred))

model_features = x.columns
joblib.dump(XGR, "xgr_model.pkl")
joblib.dump(model_features, "features_list.pkl")

XGR = joblib.load("xgr_model.pkl")
model_features = joblib.load("features_list.pkl")

st.title("🏠 Madrid Housing Price Predictor")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox('Select a Page', ['Data','Overview','Prediction'])

if page == 'Data':
    st.subheader("Cleaned Dataset")
    st.dataframe(df[['District','sq_mt_built','n_rooms','n_bathrooms','buy_price','has_parking']])
         
elif page == 'Overview':
    st.subheader('Data Overview')
    pie = df['has_parking'].map({0: 'No', 1: 'Yes'})
    fig1 = px.pie(names= pie, title= 'Parking Availability Distribution')
    st.plotly_chart(fig1)

    districts = df['District'].value_counts()
    fig2 = px.bar(districts, title= 'Distribuation of District in Madrid')
    st.plotly_chart(fig2)
    
    st.write('Top 10 Most Expensive Districts (by Average Buy Price)')
    most_expensive_districts = df.groupby('District')['buy_price'].mean().nlargest(10)
    fig3 = px.bar(most_expensive_districts, title= 'Top 10 Most Expensive Districts')
    st.plotly_chart(fig3)

    park_avg = df.groupby('has_parking')['buy_price'].mean().reset_index()
    park_avg['has_parking'] = park_avg['has_parking'].map({0: 'No', 1: 'Yes'})
    park_avg['has_parking'] = pd.Categorical(park_avg['has_parking'], categories=['No', 'Yes'], ordered=True)
    
    fig4 = px.bar(
        park_avg,
        x='has_parking',
        y='buy_price',
        title='Average Buy Price by Parking Availability',
        labels={'has_parking': 'Has Parking', 'buy_price': 'Average Buy Price (€)'},
        color='has_parking',
        color_discrete_map={'Yes': 'green', 'No': 'red'}
    )

    fig4.update_layout(showlegend=False, bargap=0.4) 
    st.plotly_chart(fig4)

elif page == 'Prediction':
    st.write("Enter property details to estimate the price in Euros.")
    # Data Input
    area = st.number_input("Area (sq. meters)", min_value=10, max_value=1000, value=100)
    rooms = st.number_input("Number of Rooms", min_value=1, max_value=24, value=3)
    baths = st.number_input("Number of Bathrooms", min_value=1, max_value=14, value=2)
    district = st.selectbox("District", [col.replace("District_", "") for col in model_features if "District_" in col])
    parking = st.selectbox("Parking Available?", ["Yes", "No"])

    input_data = np.zeros(len(model_features))
    input_data[model_features.get_loc("sq_mt_built")] = area
    input_data[model_features.get_loc("n_rooms")] = rooms
    input_data[model_features.get_loc("n_bathrooms")] = baths
    if f"District_{district}" in model_features:
        input_data[model_features.get_loc(f"District_{district}")] = 1
    if "has_parking" in model_features:
        input_data[model_features.get_loc("has_parking")] = 1 if parking == "Yes" else 0

    # Prediction
    if st.button("Predict Price 💰"):
        prediction = XGR.predict([input_data])[0]
        st.success(f"Estimated Price: **€{prediction:,.2f}**")















