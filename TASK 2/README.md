**Online Education Student Performance Analysis**
📌 Project Overview

This project analyzes an online education dataset to understand student engagement, academic performance, risk levels, and final outcomes.

The analysis uses Python, Pandas, Seaborn, Matplotlib, and Scikit-learn to explore the relationship between student engagement and academic outcomes. A Logistic Regression model is also used to examine the relationship between total student clicks and the probability of passing.

📊 Dataset

The dataset contains 32,593 student records and 14 features related to student demographics, education, engagement, performance, and outcomes.

Main Features
Feature	Description
id_student	Unique student identifier
gender	Student gender
region	Student region
highest_education	Highest level of education
studied_credits	Number of credits studied
imd_band	Socioeconomic/IMD band
total_clicks	Total number of interactions/clicks
avg_score	Student's average score
engagement_level	Student engagement category
performance_level	Student performance category
risk_level	Student risk category
pass_flag	Indicates whether the student passed
dropout_flag	Indicates whether the student dropped out
final_result	Final student outcome
Final Results

The dataset contains four final outcome categories:

Pass: 12,361 students
Withdrawn: 10,156 students
Fail: 7,052 students
Distinction: 3,024 students
🛠️ Technologies Used
Python
Pandas – Data manipulation and analysis
NumPy – Numerical operations
Matplotlib – Data visualization
Seaborn – Statistical visualization
Scikit-learn – Machine learning
Jupyter Notebook – Development environment
🔍 Project Workflow
1. Import Libraries

The project begins by importing the required Python libraries:

import numpy as np
import pandas as pd
2. Load the Dataset

The dataset is loaded using Pandas:

df = pd.read_csv("online_education_dataset.csv")
3. Explore Student Outcomes

The distribution of final results is examined using:

df["final_result"].value_counts()

The project also groups the data to examine student outcomes and engagement.

4. Analyze Engagement and Passing

The notebook calculates the average pass rate for different engagement levels:

df.groupby("engagement_level")["pass_flag"].mean()

A bar chart is created to visualize the relationship between engagement level and pass rate.

5. Handle Missing Values

Missing values in relevant columns are addressed. For example, missing total_clicks and pass_flag values are filled:

df["total_clicks"] = df["total_clicks"].fillna(0)
df["pass_flag"] = df["pass_flag"].fillna(0)
6. Logistic Regression

A Logistic Regression model is trained using total_clicks as the predictor and pass_flag as the target.

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

The feature is standardized before training:

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

The model coefficient is then examined to understand the relationship between student activity and passing.

📈 Visualizations

The project includes a visualization of Pass Rate by Engagement Level.

The visualization is generated using Seaborn:

sns.barplot(
    x="engagement_level",
    y="pass_flag",
    data=df,
    order=["low", "Medium", "High"]
)

The resulting chart is saved as:

pass_rate_by_engagement.png
🤖 Machine Learning
Model Used

Logistic Regression

Feature
total_clicks
Target
pass_flag

The purpose of this model is to examine whether student activity, measured through total clicks, is associated with the likelihood of passing.

Note: The notebook currently uses a single feature (total_clicks) for the Logistic Regression model. It is intended primarily as an analysis of the relationship between engagement/activity and passing rather than a complete student-performance prediction system.

📁 Project Structure
online-education-analysis/
│
├── education notebook(1).ipynb
├── online_education_dataset(1).csv
├── pass_rate_by_engagement.png
└── README.md

For a cleaner GitHub repository, you can rename the files to:

education_analysis.ipynb
online_education_dataset.csv
pass_rate_by_engagement.png
README.md
🎯 Key Objectives

The main objectives of this project are:

Explore student academic outcomes.
Analyze student engagement levels.
Examine the relationship between engagement and passing.
Handle missing data.
Visualize pass rates across engagement categories.
Apply Logistic Regression to study the relationship between student activity and passing.
🚀 How to Run the Project
1. Clone the repository
git clone https://github.com/your-username/online-education-analysis.git
2. Navigate to the project directory
cd online-education-analysis
3. Install the required libraries
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
4. Start Jupyter Notebook
jupyter notebook

Open:

education_analysis.ipynb

Make sure the CSV dataset is located in the same directory as the notebook.

📌 Results

The analysis shows that the dataset contains substantial variation in student outcomes and engagement.

The notebook specifically investigates how engagement level and total clicks relate to student passing outcomes. The Logistic Regression component provides a simple statistical/machine-learning approach for examining the relationship between activity and passing.

🔮 Future Improvements

The project could be extended by:

Using multiple features for prediction.
Comparing different machine-learning algorithms.
Splitting the data into training and testing sets.
Evaluating model accuracy, precision, recall, and F1-score.
Creating a confusion matrix.
Performing feature importance analysis.
Improving missing-value handling.
Creating an interactive dashboard.
Adding more visualizations for performance and risk levels.
👨‍💻 Author
   Dheen Seenivasan S
    BCA Student 
