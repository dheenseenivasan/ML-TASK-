# Electric Vehicle Price Prediction Using Linear Regression

This project demonstrates **data analysis and price prediction** using an electric vehicle dataset from India with Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## 📌 Project Overview

The project focuses on analyzing electric vehicle specifications and building a simple **Linear Regression** model to predict EV price based on driving range.

The notebook covers:

- Loading and inspecting an electric vehicle dataset
- Exploring the dataset using `head()`, `info()`, `describe()`, and `shape`
- Checking and removing missing values in `Range` and `Price`
- Selecting `Range` as the input feature
- Selecting `Price` as the target variable
- Splitting the data into training and testing sets
- Training a **Linear Regression** model
- Extracting the model slope and intercept
- Comparing actual and predicted EV prices
- Visualizing actual vs predicted prices
- Evaluating the model using MAE, MSE, RMSE, and R-squared

> **Note:** This project currently uses only `Range` to predict `Price`. Other available features such as `Power` and `Battery` are not included in the current Linear Regression model.

## 📂 Files in This Repository

| File | Description |
|---|---|
| `ev_car_India_dataset(1).csv` | Electric vehicle dataset containing 26 EV records and 6 columns |
| `EV (NOTEBOOK)(1).ipynb` | Notebook for EV data exploration and price prediction |
| `README.md` | Project documentation |

## 📊 Dataset

The dataset contains **26 electric vehicle records** and **6 columns**.

Important columns include:

- `Brand` – EV manufacturer/brand
- `Model` – Electric vehicle model
- `Price` – Price value provided in the dataset
- `Range` – Driving range value provided in the dataset
- `Power` – Vehicle power value provided in the dataset
- `Battery` – Battery capacity value provided in the dataset

### Dataset Shape

```text
Rows: 26
Columns: 6
```

### Dataset Summary

| Statistic | Price | Range | Power | Battery |
|---|---:|---:|---:|---:|
| Count | 26 | 26 | 26 | 26 |
| Mean | 28.77 | 443.42 | 191.53 | 52.68 |
| Minimum | 3.25 | 175.00 | 16.00 | 12.60 |
| Maximum | 75.00 | 663.00 | 503.00 | 90.00 |

## 🧹 Data Cleaning

The notebook checks for missing values in the two columns used for modeling:

```python
df = df.dropna(subset=["Range","Price"])
```

The remaining missing values are checked using:

```python
df[["Range","Price"]].isnull().sum()
```

The dataset used in the notebook contains **26 records after this preprocessing step**.

## 📈 Feature and Target Selection

The notebook uses:

### Feature

```python
X = df[["Range"]]
```

`Range` is used as the independent variable.

### Target

```python
y = df[["Price"]]
```

`Price` is used as the dependent variable.

The current model therefore investigates the relationship between **EV range and price**.

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using an 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

```text
Training records: 20
Testing records: 6
```

## 🤖 Linear Regression

Scikit-learn's `LinearRegression` is used:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

The model calculates a slope and intercept:

```python
print("Slope", model.coef_[0])
print("Intercept", model.intercept_)
```

For the current dataset and train-test split, the model produces approximately:

```text
Slope: 0.1414
Intercept: -34.3961
```

The fitted relationship can therefore be represented approximately as:

```text
Predicted Price = 0.1414 × Range - 34.3961
```

## 🔮 Price Prediction

Predictions are generated for the test dataset:

```python
y_pred = model.predict(X_test)
```

A comparison DataFrame is then created:

```python
comparison_df = pd.DataFrame({
    "Actual Price": y_test.values.ravel(),
    "Predicted Price": y_pred.ravel()
})
```

This allows the actual and predicted price values to be compared.

## 📊 Actual vs Predicted Visualization

The notebook creates a scatter plot comparing actual and predicted EV prices:

```python
plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted EV Price")
plt.show()
```

The diagonal reference line represents where predicted values would equal actual values.

## 📏 Model Evaluation

The notebook evaluates the Linear Regression model using four metrics:

### Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
```

Result:

```text
MAE: 17.5081
```

### Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

Result:

```text
MSE: 410.5502
```

### Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mse)
```

Result:

```text
RMSE: 20.2620
```

### R-squared

```python
r2 = r2_score(y_test, y_pred)
```

Result:

```text
R-squared: 0.2179
```

These results are based on the notebook's 80/20 split with `random_state=42`.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Open the notebook

Open:

```text
EV (NOTEBOOK)(1).ipynb
```

You can run the notebook using Jupyter Notebook, JupyterLab, or Google Colab.

## ⚠️ Dataset File Path

The notebook currently contains:

```python
df = pd.read_csv("ev_car_India_dataset.csv")
```

The uploaded dataset is named:

```text
ev_car_India_dataset(1).csv
```

If you keep the uploaded filename in your GitHub repository, update the notebook path to:

```python
df = pd.read_csv("ev_car_India_dataset(1).csv")
```

Alternatively, rename the CSV file to:

```text
ev_car_India_dataset.csv
```

and keep the existing notebook code unchanged.

## 📈 Future Improvements

Possible next steps for this project include:

- Using `Range`, `Power`, and `Battery` together for price prediction
- Comparing Linear Regression with other machine learning algorithms
- Performing exploratory data analysis using the EV specifications
- Creating correlation heatmaps
- Visualizing price relationships with Range, Power, and Battery
- Increasing the dataset size for more reliable model evaluation
- Performing cross-validation
- Improving model performance through feature selection
- Evaluating additional regression models
- Building an interactive EV price prediction application

## 👤 Author

**Your Name**

BCA Student | Data Analysis | Machine Learning

---

⭐ If you find this project useful, feel free to star the repository.
