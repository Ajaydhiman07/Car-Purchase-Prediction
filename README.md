# 🚗 Car Purchase Prediction using Machine Learning

This project predicts a customer's **car purchase amount** based on their demographic and financial information using various **machine learning models**.  
The goal is to analyze different models, evaluate their performance, and select the most accurate one for predicting car purchase behavior.

---

## 📁 Project Structure
Car-Purchase-Prediction/
│
├── Car_Purchasing_Data.csv # Dataset used for training and testing
├── Car_Sale_Prediction.ipynb # Main Jupyter Notebook with ML workflow
├── README.md # Project documentation (you are reading this)
└── requirements.txt # List of required Python packages (optional)


---

## 📊 Dataset Description

The dataset contains **500 records** with the following columns:

| Feature | Description |
|----------|-------------|
| Customer Name | Name of the customer |
| Customer e-mail | Email of the customer |
| Country | Country of residence |
| Gender | Gender (Male/Female) |
| Age | Age of the customer |
| Annual Salary | Annual income (in USD) |
| Credit Card Debt | Amount of debt on credit card |
| Net Worth | Total net worth of the customer |
| Car Purchase Amount | Target variable – amount spent on car purchase |

---

## ⚙️ Workflow Overview

1. **Data Loading and Cleaning**  
   - Load the dataset using pandas.  
   - Handle missing values and remove irrelevant columns like *Customer Name* and *Customer e-mail*.

2. **Exploratory Data Analysis (EDA)**  
   - Visualize relationships between income, debt, and car purchase amount.  
   - Check data distribution, correlations, and outliers using Seaborn and Matplotlib.

3. **Feature Engineering**  
   - Encode categorical variables like Gender.  
   - Drop unused columns (like names and emails).  
   - Scale numerical features using StandardScaler.

4. **Model Training**  
   - Split the dataset into training and testing sets (75/25).  
   - Train multiple models:
     - Linear Regression  
     - Decision Tree Regressor  
     - Random Forest Regressor  
     - Gradient Boosting Regressor  
     - Support Vector Regressor (SVR)

5. **Model Evaluation**  
   - Compare model performances using metrics:
     - MAE (Mean Absolute Error)  
     - RMSE (Root Mean Squared Error)  
     - R² (Coefficient of Determination)

6. **Visualization**  
   - Scatter plots for `Annual Salary` vs `Car Purchase Amount` and others.  
   - Actual vs Predicted comparison plots.  
   - Regression lines for better insight into model behavior.

7. **Prediction on New Data**  
   - Provide an example input (new customer data).  
   - Predict the potential car purchase amount.

---

## 🧠 Machine Learning Models Used

| Model | Description |
|--------|-------------|
| **Linear Regression** | Establishes a simple linear relationship between features and target |
| **Decision Tree Regressor** | Tree-based non-linear model |
| **Random Forest Regressor** | Ensemble of trees improving overfitting and accuracy |
| **Gradient Boosting Regressor** | Boosting technique with excellent accuracy |
| **Support Vector Regressor (SVR)** | Non-linear kernel-based regression |

---

## 📈 Sample Visualizations

- Age vs Car Purchase Amount  
- Annual Salary vs Car Purchase Amount  
- Actual vs Predicted Car Purchase Amount  
- Pairplot of Numerical Features  
- Regression Lines for Each Feature

---


## 💻 Installation and Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Ajaydhiman07/Car-Purchase-Prediction.git
cd Car-Purchase-Prediction

2️⃣ Install dependencies
If you don’t have the required libraries, install them via pip:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost

3️⃣ Run the Jupyter Notebook
jupyter notebook Car_Sale_Prediction.ipynb
