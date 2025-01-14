# Houses in Madrid - Data Analysis and Prediction

This project involves analyzing and predicting house prices in Madrid. The dataset contains various features about properties such as their size, price, number of rooms, bathrooms, and the district they are located in. The analysis and predictions aim to understand the factors that influence house prices and to predict the price based on specific characteristics.

## Project Overview

- **Dataset**: The dataset contains information about properties in Madrid, including columns such as:
  - `district`: The district in which the property is located.
  - `sq_mt_built`: The built area of the property (in square meters).
  - `sq_mt_useful`: The useful area of the property (in square meters).
  - `n_rooms`: The number of rooms in the property.
  - `n_bathrooms`: The number of bathrooms in the property.
  - `rent_price`: The rent price of the property (if available).
  - `buy_price`: The buying price of the property.
  - `buy_price_by_area`: The price per square meter for the property.
  - `built_year`: The year the property was built.
  - `parking_price`: The price for parking (if applicable).

- **Objective**: The goal of this project is to analyze the data, identify patterns, and build prediction models that estimate house prices based on different features such as area, number of rooms, and location.

## Data Analysis and Insights

The analysis includes the following steps:

1. **Exploratory Data Analysis (EDA)**:
   - Visualize distributions of numerical columns.
   - Analyze correlations between features.
   - Handle missing or incorrect data, such as replacing missing values in `n_bathrooms`.

2. **Price Predictions**:
   - Use multiple linear regression models to predict property prices based on features like area and number of bathrooms.
   - Evaluate model performance and interpret results to understand the key factors that influence house prices in Madrid.

3. **District-Level Analysis**:
   - Analyze how the house prices vary across different districts in Madrid.
   - Visualize average prices by district to identify the most expensive and affordable areas.
