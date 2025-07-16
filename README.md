# 🚲 Forecasting Bike Rentals with Tree-Based Models and OLS

This project builds a regression pipeline to forecast hourly bike rental demand using weather, time, and calendar features. It benchmarks multiple models (OLS, CART, Random Forest, XGBoost) and evaluates the impact of hyperparameter tuning and model complexity on predictive accuracy.

---

## 🎯 Objective

Develop a regression model to accurately predict the number of bike rentals using structured data, while comparing baseline OLS with tree-based models. Use advanced evaluation metrics and tuning strategies to optimize model performance.

---

## 📦 Dataset

- File: `rentals_weather.csv`
- Features:
  - **Time:** `month`, `day`, `hour`, `day_of_week`, `weekend`
  - **Weather:** `temp`, `humidity`, `windspeed`, etc.
  - **Target:** `rentals` (hourly rental count)

---

## 🧠 Models Used

- OLS (Linear Regression)
- CART (`DecisionTreeRegressor`)
- Random Forest
- XGBoost

---

## ⚙️ Methodology

- Data cleaning + categoricals casting
- 70/30 train-test split
- Manual feature selection
- Model tuning using:
  - `GridSearchCV`
  - `RandomizedSearchCV`
- Custom metric: **Out-of-sample R² (OSR2)**

---

## 📈 Key Results

- >45% RMSE reduction over OLS using tuned tree-based models
- XGBoost outperformed CART and RF in both OSR2 and RMSE
- Ensemble methods yielded lower variance on unseen data
- RandomizedSearchCV achieved comparable results to GridSearchCV with fewer trials

---

## 📁 Files

- `CART_Regression.ipynb`: Full pipeline, comparisons, plots, metrics
- `rentals_weather.csv`: Data used for training and testing

---

## 🛠️ Stack

- Python, Pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib
