🧹 Data Cleaning & PreprocessingThe notebook verifies that there are zero missing values in the dataset:   Pythondf.isnull().sum()
   Output:   PlaintextBrand      0
Model      0
Price      0
Range      0
Power      0
Battery    0
dtype: int64
   Feature Transformation PipelineTo handle mixed data types without data leakage, a ColumnTransformer is created:   Pythonfrom sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical = ["Brand", "Model"]
numerical = ["Range", "Power", "Battery"]

preprocessor = ColumnTransformer([
    ("Categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("Numerical", StandardScaler(), numerical)
])
   OneHotEncoder encodes categorical brand and model indicators.   StandardScaler standardizes numeric features to have zero mean and unit variance.   📈 Feature and Target SelectionThe dataset is partitioned into predictors and the target variable:   Features ($X$)PythonX = df.drop("Price", axis=1)
   Includes Brand, Model, Range, Power, and Battery.   Target ($y$)Pythony = df["Price"]
   The vehicle price is used as the continuous target variable.   ✂️ Train-Test SplitThe data is split into an 80% training set and a 20% test set:   PythonX_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
[cite: 3]This results in:PlaintextTraining records: 20
Testing records: 6
🤖 Ridge Regression PipelineThe preprocessing step and Ridge regression estimator are linked through a unified Pipeline[cite: 3]:Pythonfrom sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

model = Pipeline([
    ("Preprocessing", preprocessor),
    ("Ridge", Ridge(alpha=100))
])

model.fit(X_train, y_train)
[cite: 3]L2 regularization helps mitigate multicollinearity and prevents overfitting on small datasets[cite: 3].🔮 PredictionsPredictions are computed for both training and testing partitions[cite: 3]:Pythontrain_pred = model.predict(X_train)
test_pred = model.predict(X_test)
[cite: 3]📏 Model EvaluationThe notebook evaluates model performance on training and unseen test data[cite: 3]:Evaluation CodePythonfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

train_mae = mean_absolute_error(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_r2 = r2_score(y_train, train_pred)

test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)
[cite: 3]Results Summary ($\alpha = 100$)Dataset SplitMAERMSER-squared (R2)Train Set13.004214.99660.4683Test Set16.214618.56490.3434[cite: 3]🛠️ Technologies UsedPython[cite: 3]Pandas[cite: 3]NumPy[cite: 3]Matplotlib[cite: 3]Seaborn[cite: 3]Scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler, Ridge)[cite: 3]Jupyter Notebook / Google Colab[cite: 3]🚀 How to Run1. Clone the repositoryBashgit clone <your-repository-url>
cd <your-repository-folder>
2. Install dependenciesBashpip install pandas numpy matplotlib seaborn scikit-learn
3. Open the notebookOpen:Plaintextridge regression.ipynb
[cite: 3]You can execute cells interactively using Jupyter Notebook, JupyterLab, or Google Colab[cite: 3].⚠️ Dataset File PathThe notebook reads the dataset using[cite: 3]:Pythondf = pd.read_csv("/content/ev_car_India_dataset.csv")
[cite: 3]When running locally, adjust the file path to match your directory[cite: 3]:Pythondf = pd.read_csv("ev_car_India_dataset - ev_car_India_dataset.csv")
or rename the CSV file to ev_car_India_dataset.csv and keep the notebook code unchanged[cite: 3].📈 Future ImprovementsPerform cross-validation and hyperparameter grid search (GridSearchCV) over alphas = [0.01, 0.1, 1, 10, 100]Compare Ridge regression with Lasso, ElasticNet, and Tree-based Ensembles (Random Forest, XGBoost)Incorporate charging speed and warranty information into the feature spaceExpand the dataset with more electric car models sold in the Indian marketBuild an interactive Streamlit or Flask web application for price estimation👤 AuthorDheen SeenivasanBCA Student | Data Analysis | Machine Learning⭐ If you find this project useful, feel free to star the repository.
