# Customer Segmentation Using RFM Analysis & K-Means Clustering

This project demonstrates **customer segmentation** using **RFM (Recency, Frequency, Monetary) analysis** and **K-Means clustering** with Python, Pandas, Scikit-learn, and Matplotlib.

## 📌 Project Overview

The project focuses on cleaning sales transaction data, creating RFM features for each customer, scaling the features, and grouping customers into clusters using K-Means clustering.

The notebook covers:

- Loading and inspecting a sales dataset
- Checking for missing values
- Removing transactions without a `Customer ID`
- Removing cancelled invoices
- Removing invalid transactions with non-positive quantity or price
- Removing duplicate records
- Calculating total transaction amount
- Creating **Recency, Frequency, and Monetary (RFM)** features
- Standardizing RFM features using `StandardScaler`
- Finding a suitable number of clusters using the **Elbow Method**
- Applying **K-Means clustering**
- Evaluating clusters using **Silhouette Score**
- Evaluating clusters using the **Davies-Bouldin Index**
- Visualizing customer clusters using Frequency and Monetary values
- Creating a cluster summary using average RFM values

> **Note:** This project focuses on customer segmentation using unsupervised machine learning. It does not contain a supervised prediction model.

## 📂 Files in This Repository

| File | Description |
|---|---|
| `sales dataset(1).xlsx` | Sales transaction dataset |
| `Sales analysis notebook(1).ipynb` | Notebook containing data cleaning, RFM analysis, clustering, evaluation, and visualization |
| `0001(8).jpeg` | Customer cluster visualization |
| `README.md` | Project documentation |

## 📊 Dataset

The sales dataset contains **525,461 transaction records** and **8 columns** before preprocessing.

Important columns include:

- `Invoice` – Invoice number for a transaction
- `StockCode` – Product/item code
- `Description` – Product description
- `Quantity` – Number of items purchased
- `InvoiceDate` – Date and time of the transaction
- `Price` – Unit price of the product
- `Customer ID` – Unique customer identifier
- `Country` – Customer country

### Dataset Shape

```text
Rows: 525,461
Columns: 8
```

## 🧹 Data Cleaning

The notebook performs several preprocessing steps before creating customer-level features.

### 1. Remove missing Customer IDs

Transactions without a customer identifier are removed:

```python
df = df.dropna(subset=["Customer ID"])
```

### 2. Remove cancelled invoices

Invoices beginning with `C` are treated as cancelled transactions and removed:

```python
df = df[~df["Invoice"].astype(str).str.startswith("C")]
```

### 3. Remove invalid transactions

Transactions with zero or negative quantity or price are removed:

```python
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
```

### 4. Remove duplicate records

Duplicate transaction records are removed:

```python
df = df.drop_duplicates()
```

After preprocessing, the notebook contains **400,916 transaction records** and **4,312 unique customers**.

## 💰 Total Amount Calculation

A new `TotalAmount` column is created by multiplying quantity by price:

```python
df["TotalAmount"] = df["Quantity"] * df["Price"]
```

This value is used to calculate the Monetary component of RFM analysis.

## 📈 RFM Analysis

RFM analysis is used to describe customer purchasing behavior using three features:

### Recency

Measures how recently a customer made a purchase.

```python
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
```

Recency is calculated as the number of days since the customer's most recent purchase.

### Frequency

Measures how many unique invoices a customer has made:

```python
Frequency=("Invoice", "nunique")
```

### Monetary

Measures the total amount spent by the customer:

```python
Monetary=("TotalAmount", "sum")
```

The three features are combined into an RFM table:

```python
rfm = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate",
             lambda x: (reference_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalAmount", "sum")
)
```

## ⚖️ Feature Scaling

The RFM features are standardized using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

features = ["Recency", "Frequency", "Monetary"]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm[features])
```

Standardization places the RFM features on a comparable scale before clustering.

## 🔵 K-Means Clustering

K-Means clustering is used to group customers based on their RFM characteristics.

The notebook first tests cluster counts from 2 to 10 using the Elbow Method:

```python
inertia = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)
    inertia.append(model.inertia_)
```

The final notebook applies **2 clusters**:

```python
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
```

## 📏 Cluster Evaluation

### Silhouette Score

The notebook calculates the Silhouette Score:

```python
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(
    rfm_scaled,
    rfm["Cluster"]
)
```

The resulting score is approximately:

```text
Silhouette Score: 0.9314
```

### Davies-Bouldin Index

The notebook also calculates the Davies-Bouldin Index:

```python
from sklearn.metrics import davies_bouldin_score

db_index = davies_bouldin_score(
    rfm_scaled,
    rfm["Cluster"]
)
```

The resulting value is approximately:

```text
Davies-Bouldin Index: 0.5834
```

## 📊 Customer Cluster Visualization

The customer clusters are visualized using **Frequency** on the X-axis and **Monetary** on the Y-axis:

```python
plt.scatter(
    rfm["Frequency"],
    rfm["Monetary"],
    c=rfm["Cluster"]
)

plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Customer Clusters")
plt.show()
```

The visualization shows two customer groups based on their purchasing behavior.

## 📋 Cluster Summary

The notebook calculates the average RFM values for each cluster:

```python
cluster_summary = rfm.groupby("Cluster")[
    ["Recency", "Frequency", "Monetary"]
].mean()

print(cluster_summary)
```

The resulting averages are approximately:

| Cluster | Recency | Frequency | Monetary |
|---|---:|---:|---:|
| 0 | 91.39 | 4.17 | 1,718.13 |
| 1 | 4.73 | 114.27 | 128,051.99 |

The notebook contains **4,312 customers** in total, with 4,301 customers assigned to Cluster 0 and 11 customers assigned to Cluster 1.

These values describe the clusters based on the RFM features and are not manually assigned customer labels.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab
- Excel dataset

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

### 3. Open the notebook

Open:

```text
Sales analysis notebook(1).ipynb
```

You can run the notebook using Jupyter Notebook, JupyterLab, or Google Colab.

## ⚠️ Dataset File Path

The notebook currently contains a Colab path:

```python
pd.read_excel("/content/sales2.xlsx")
```

If the Excel file in your GitHub repository has a different name, update the path accordingly. For example:

```python
df = pd.read_excel("sales dataset(1).xlsx")
```

## 📈 Future Improvements

Possible next steps for this project include:

- Creating more detailed customer segment descriptions
- Comparing different numbers of clusters
- Performing additional exploratory data analysis
- Visualizing all RFM dimensions
- Analyzing customer segments by country
- Studying purchasing trends over time
- Creating an interactive customer segmentation dashboard
- Developing customer-specific marketing strategies based on the segments

## 👤 Author

**Your Name**

BCA Student | Data Analysis | Machine Learning

---

⭐ If you find this project useful, feel free to star the repository.
