# Household Energy Consumption Prediction Using Polynomial Regression

This project demonstrates **data analysis and energy consumption prediction** using a residential power consumption dataset with Python, Pandas, Matplotlib, and Scikit-learn.

## 📌 Project Overview

The project focuses on analyzing household energy usage patterns and training a **Polynomial Regression (Degree 2)** model to predict total daily energy consumption based on household size, ambient temperature, and peak-hour electricity usage.

The notebook covers:

- Loading and inspecting the household energy consumption dataset
- Exploring the dataset using `shape`, `head()`, `info()`, `describe()`, and `columns`
- Checking and handling missing values using `isnull().sum()` and `dropna()`
- Selecting `Household_Size`, `Avg_Temperature_C`, and `Peak_Hours_Usage_kWh` as feature variables
- Selecting `Energy_Consumption_kWh` as the target variable
- Splitting the data into training and testing sets (80/20 split)
- Transforming input features using `PolynomialFeatures(degree=2)`
- Training a `LinearRegression` model on the polynomial features
- Comparing actual and predicted energy consumption values
- Visualizing actual vs. predicted consumption using a scatter plot
- Evaluating the model using MAE, MSE, RMSE, and R-squared ($R^2$)

> **Note:** This project currently trains on `Household_Size`, `Avg_Temperature_C`, and `Peak_Hours_Usage_kWh`. Other columns such as `Household_ID`, `Date`, and `Has_AC` are not included in the feature matrix.

## 📂 Files in This Repository

| File | Description |
|---|---|
| `household_energy_consumption - household_energy_consumption.csv` | Dataset containing 90,000 household energy records and 7 columns |
| `Ml_task4.ipynb` | Google Colab / Jupyter notebook for data exploration and polynomial regression modeling |
| `README.md` | Project documentation |

## 📊 Dataset

The dataset contains **90,000 records** and **7 columns**.

Important columns include:

- `Household_ID` – Unique identifier for each household
- `Date` – Date of the observation
- `Household_Size` – Number of occupants in the household
- `Avg_Temperature_C` – Average daily ambient temperature (°C)
- `Has_AC` – Air conditioning status (`Yes` or `No`)
- `Peak_Hours_Usage_kWh` – Energy consumed during peak hours (kWh)
- `Energy_Consumption_kWh` – Total daily energy consumption (target variable, kWh)

### Dataset Shape

```text
Rows: 90000
Columns: 7
