# 🔍 Advanced Regression Modeling with Scikit-learn

This project is a deep dive into modern regression techniques using Python and Scikit-learn. It covers practical implementation and analysis of various models including linear regression, polynomial regression, regularization methods like Ridge and Lasso, and also custom pipelines and estimators.

The goal is to build a solid, extensible foundation for machine learning workflows involving structured data, model tuning, and evaluation — all with clean, modular, and reusable code.

---

## 📁 Dataset

- **File**: `data2_200x30.csv`
- **Samples**: 200 rows
- **Features**: 30 anonymous input features + 1 target output
- **Preprocessing Rule**: No feature engineering (no dealing with outliers, missing values, or duplicates).
- **Scaling**: MinMaxScaler by default, with options to test StandardScaler or no scaling.

---

## ⚙️ Tech Stack

- Python 3.10+
- Scikit-learn
- NumPy
- Matplotlib
- Argparse (for CLI interface)

---

## 📌 Key Features

- ✅ Modular code for training and evaluating regression models
- ✅ CLI interface using `argparse` for easy experimentation
- ✅ Model evaluation using RMSE
- ✅ Extensive use of Scikit-learn features like Pipelines, GridSearchCV, and custom estimators
- ✅ Visualizations for comparison and analysis
- ✅ Organized output folders per task
- ✅ Custom utilities for monomial polynomial features
- ✅ Feature selection using Lasso + Ridge combination

---

## 🧪 Tasks Overview

### 1. **Simple Linear Regression (Baseline)**
Train a baseline model using unregularized linear regression with all features. Evaluate using RMSE.

### 2. **Polynomial Regression (Cross Features)**
Use PolynomialFeatures from sklearn for degrees [1, 2, 3, 4]. Compare overfitting using visualized RMSE.

### 3. **Polynomial Regression (Monomials Only)**
Generate polynomial features without cross-terms (e.g., x, x², x³). Evaluate model generalization.

### 4. **Individual Feature Analysis**
Train polynomial models using individual features and visualize comparative RMSE for selected degrees.

### 5. **Ridge Regression + Cross Validation**
Apply ridge regression with polynomial features and perform model selection via GridSearchCV over alpha.

### 6. **Lasso-Based Feature Selection**
Use Lasso to select up to 10 key features, then apply Ridge regression on selected features only.

### 7. **Regularized Normal Equation**
Implement a `NormalEquations` class supporting regularization, following closed-form solution.

### 8. **Regularized Gradient Descent**
Extend gradient descent implementation to support L2 regularization (lambda as a hyperparameter).

### 9. **Grid Search Pipeline (v1)**
Use Pipeline + GridSearchCV to explore combinations of:
- Polynomial degrees
- Ridge alphas
- fit_intercept

### 10. **Custom Estimator: Polynomial + Ridge**
Create a custom class `PolynomialRegression(BaseEstimator)` that transforms and fits a Ridge model based on user parameters.

---

## 📊 Output Structure

All results and plots are stored per task under an `outputs/` folder:

