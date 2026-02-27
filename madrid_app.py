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

df['subtitle'] = df['subtitle'].str.split(',', expand= True)[0]

df = df.rename(columns= {'subtitle' : 'District'})

df['has_parking'] = df['has_parking'].astype(int)
         
df1 = pd.get_dummies(df, columns=['District'], drop_first=True)

x = df1[['sq_mt_built', 'n_rooms', 'n_bathrooms', 'has_parking', 'District_Abrantes', 'District_Acacias', 'District_Adelfas', 'District_Alameda de Osuna', 'District_Almagro', 'District_Almendrales', 'District_Aluche', 'District_Ambroz', 'District_Apóstol Santiago', 'District_Arapiles', 'District_Aravaca', 'District_Arganzuela', 'District_Argüelles', 'District_Arroyo del Fresno', 'District_Atalaya', 'District_Barajas', 'District_Barrio de Salamanca', 'District_Bellas Vistas', 'District_Bernabéu-Hispanoamérica', 'District_Berruguete', 'District_Buena Vista', 'District_Butarque', 'District_Campamento', 'District_Campo de las Naciones-Corralejos', 'District_Canillas', 'District_Carabanchel', 'District_Casa de Campo', 'District_Casco Histórico de Barajas', 'District_Casco Histórico de Vallecas', 'District_Casco Histórico de Vicálvaro', 'District_Castellana', 'District_Castilla', 'District_Centro', 'District_Chamartín', 'District_Chamberí', 'District_Chopera', 'District_Chueca-Justicia', 'District_Ciudad Jardín', 'District_Ciudad Lineal', 'District_Ciudad Universitaria', 'District_Colina', 'District_Comillas', 'District_Concepción', 'District_Conde Orgaz-Piovera', 'District_Costillares', 'District_Cuatro Caminos', 'District_Cuatro Vientos', 'District_Cuzco-Castillejos', 'District_Delicias', 'District_El Cañaveral - Los Berrocales', 'District_El Pardo', 'District_El Plantío', 'District_El Viso', 'District_Ensanche de Vallecas - La Gavia', 'District_Entrevías', 'District_Estrella', 'District_Fontarrón', 'District_Fuencarral', 'District_Fuente del Berro', 'District_Fuentelarreina', 'District_Gaztambide', 'District_Goya', 'District_Guindalera', 'District_Horcajo', 'District_Hortaleza', 'District_Huertas-Cortes', 'District_Ibiza', 'District_Imperial', 'District_Jerónimos', 'District_La Paz', 'District_Las Tablas', 'District_Latina', 'District_Lavapiés-Embajadores', 'District_Legazpi', 'District_Lista', 'District_Los Cármenes', 'District_Los Rosales', 'District_Los Ángeles', 'District_Lucero', 'District_Malasaña-Universidad', 'District_Marroquina', 'District_Media Legua', 'District_Mirasierra', 'District_Moncloa', 'District_Montecarmelo', 'District_Moratalaz', 'District_Moscardó', 'District_Niño Jesús', 'District_Nueva España', 'District_Nuevos Ministerios-Ríos Rosas', 'District_Numancia', 'District_Opañel', 'District_Orcasitas', 'District_Pacífico', 'District_Palacio', 'District_Palomas', 'District_Palomeras Bajas', 'District_Palomeras sureste', 'District_Palos de Moguer', 'District_Pau de Carabanchel', 'District_Pavones', 'District_Peñagrande', 'District_Pilar', 'District_Pinar del Rey', 'District_Portazgo', 'District_Pradolongo', 'District_Prosperidad', 'District_Pueblo Nuevo', 'District_Puente de Vallecas', 'District_Puerta Bonita', 'District_Puerta del Ángel', 'District_Quintana', 'District_Recoletos', 'District_Retiro', 'District_San Andrés', 'District_San Cristóbal', 'District_San Diego', 'District_San Fermín', 'District_San Isidro', 'District_San Juan Bautista', 'District_San Pascual', 'District_Sanchinarro', 'District_Santa Eugenia', 'District_Sol', 'District_Tetuán', 'District_Timón', 'District_Trafalgar', 'District_Tres Olivos - Valverde', 'District_Usera', 'District_Valdeacederas', 'District_Valdebebas - Valdefuentes', 'District_Valdebernardo - Valderribas', 'District_Valdemarín', 'District_Valdezarza', 'District_Vallehermoso', 'District_Ventas', 'District_Ventilla-Almenara', 'District_Vicálvaro', 'District_Villa de Vallecas', 'District_Villaverde', 'District_Vinateros', 'District_Virgen del Cortijo - Manoteras', 'District_Vista Alegre', 'District_Zofío', 'District_Águilas']]
y = df1['buy_price']
y = np.array(y)

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









