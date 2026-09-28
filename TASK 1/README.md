# Customer Churn Data Preprocessing & Feature Scaling

This project demonstrates basic **data preprocessing** and **feature scaling** using a customer churn dataset with Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## 📌 Project Overview

The project focuses on preparing customer churn data for further analysis or machine learning.

The notebooks cover:

- Loading and inspecting a customer churn dataset
- Checking for missing values
- Removing columns containing missing values
- Handling missing values in the `Age` column using the mean
- Selecting numerical features for preprocessing
- Applying **Min-Max Scaling**
- Applying **Standard Scaling (Z-score normalization)** to an example array
- Basic dataset exploration using `head()`, `tail()`, `info()`, and `describe()`

> **Note:** This project currently focuses on data preprocessing and feature scaling. It does not contain a trained churn prediction model or model evaluation.

## 📂 Files in This Repository

| File | Description |
|---|---|
| `churn model.csv` | Customer churn dataset containing 10,000 records and 14 columns |
| `Churn_Notebook.ipynb` | Notebook for data inspection and missing-value handling |
| `FeatureScaling_Notebook.ipynb` | Notebook demonstrating feature selection and scaling |
| `README.md` | Project documentation |

## 📊 Dataset

The dataset contains **10,000 customer records** and **14 columns**.

Important columns include:

- `CreditScore` – Customer credit score
- `Geography` – Customer location
- `Gender` – Customer gender
- `Age` – Customer age
- `Tenure` – Number of years with the bank
- `Balance` – Account balance
- `NumOfProducts` – Number of bank products used
- `HasCrCard` – Whether the customer has a credit card
- `IsActiveMember` – Whether the customer is an active member
- `EstimatedSalary` – Estimated customer salary
- `Exited` – Churn/exit indicator

### Dataset Shape

```text
Rows: 10,000
Columns: 14
```

## 🧹 Missing Value Handling

The `Churn_Notebook.ipynb` notebook checks missing values using:

```python
df.isnull().sum()
```

It also demonstrates two approaches:

### 1. Removing columns with missing values

```python
updated_df = df.dropna(axis=1)
```

### 2. Filling missing Age values

The mean age is calculated and used to fill missing values:

```python
updated_df['Age'] = updated_df['Age'].fillna(df['Age'].mean())
```

## ⚖️ Feature Scaling

The `FeatureScaling_Notebook.ipynb` notebook selects:

```python
new_df = pd.DataFrame(df, columns=['Age', 'Tenure'])
```

Missing `Age` values are filled using the mean:

```python
new_df['Age'] = new_df['Age'].fillna(new_df['Age'].mean())
```

### Min-Max Scaling

Scikit-learn's `MinMaxScaler` is used:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(new_df)
```

Min-Max Scaling transforms numerical values to a common range, typically between 0 and 1.

### Standard Scaling

The notebook also demonstrates `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_arr_ss = scaler.fit_transform(x_array)
```

Standard Scaling transforms values based on their mean and standard deviation.

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

### 3. Open the notebooks

Open either:

```text
Churn_Notebook.ipynb
```

or

```text
FeatureScaling_Notebook.ipynb
```

You can run them using Jupyter Notebook, JupyterLab, or Google Colab.

## ⚠️ Dataset File Path

The uploaded CSV is named:

```text
churn model.csv
```

The notebooks currently contain Colab paths with different filenames:

```python
pd.read_csv("/content/Churn_modelling.csv")
```

and

```python
pd.read_csv('/content/Churn_Modelling1.csv')
```

When running the notebooks in this repository, update the `pd.read_csv()` path to match the CSV filename in your GitHub repository, for example:

```python
df = pd.read_csv("churn model.csv")
```

## 📈 Future Improvements

Possible next steps for this project include:

- Exploratory Data Analysis (EDA)
- Data visualization
- Encoding categorical variables
- Feature selection
- Train/test splitting
- Building a machine learning model
- Customer churn prediction
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix

## 👤 Author

**Dheen Seenivasan**

BCA Student | Graphic Designer | Video Editor

---

⭐ If you find this project useful, feel free to star the repository.
