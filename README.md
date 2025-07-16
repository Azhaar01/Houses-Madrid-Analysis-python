# Houses in Madrid - Data Analysis and Prediction

This project involves analyzing and predicting house prices in Madrid. The dataset contains various features about properties such as their size, price, number of rooms, bathrooms, and the district they are located in. The analysis and predictions aim to understand the factors that influence house prices and to predict the price based on specific characteristics.

## Project Overview

- **Dataset**: The dataset contains information about properties in Madrid, including columns such as:
  - `district`: The district in which the property is located.
  - `sq_mt_built`: The built area of the property (in square meters).
  - `n_rooms`: The number of rooms in the property.
  - `n_bathrooms`: The number of bathrooms in the property.
  - `buy_price`: The buying price of the property.
  - `buy_price_by_area`: The price per square meter for the property.
  - `has_parking`: Indicates whether the property includes parking (0 = No, 1 = Yes)

- **Objective**: The goal of this project is to analyze the data, identify patterns, and build prediction models that estimate house prices based on different features such as area, number of rooms, and location.

## Data Analysis

The analysis includes the following steps:

1. **Data Exploration:** Understanding the structure, distribution, and key statistics of the dataset.  
2. **Data Cleaning:** Handling missing values and inconsistent formats.  
3. **Feature Engineering:** Creating new features and encoding categorical variables.  
4. **Visualization:** Generating plots to uncover relationships and trends in the data.  
5. **Modeling:** Training predictive models such as Linear Regression, Random Forest, and XGboost to estimate housing prices.  
6. **Evaluation:** Assessing model performance using metrics like R², MAE, and RMSE.  
7. **Deployment:** Creating a simple Streamlit web app to allow users to interactively explore the data and make predictions.
   
