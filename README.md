# Car-Purchase-Amount-Prediction

## Analysis Pipeline

This project follows a complete statistical machine learning workflow to predict customer **Car Purchase Amount** using financial variables.

### 1. Data Loading and Initial Inspection

The dataset was loaded from `Car_Purchasing_Data.csv`. The data contains customer-level variables including `Gender`, `Age`, `Annual Salary`, `Credit Card Debt`, `Net Worth`, and the target variable `Car Purchase Amount`. Initial inspection confirmed that the dataset had no missing values. Non-predictive identifier columns such as `Customer Name`, `Customer e-mail`, and `Country` were removed before modeling.

### 2. Exploratory Data Analysis

Exploratory data analysis was performed to understand feature distributions and relationships between predictors and the target variable. Scatterplots and regression plots were used to examine relationships such as:

* `Age` vs `Annual Salary`
* `Age` vs `Car Purchase Amount`
* `Annual Salary` vs `Car Purchase Amount`
* `Net Worth` vs `Car Purchase Amount`

The relationship between `Age` and `Annual Salary` appeared weak, while `Age`, `Annual Salary`, and `Net Worth` showed stronger relationships with `Car Purchase Amount`. Pearson correlation analysis showed that `Age` had a moderate positive correlation with `Car Purchase Amount`.

### 3. Inferential Statistical Analysis

Inferential statistics were added to validate whether observed relationships were statistically meaningful. The analysis included:

* Pearson correlation tests for continuous predictors
* Hypothesis testing for feature-target relationships
* OLS regression summary with coefficients, p-values, confidence intervals, and F-statistic
* Multicollinearity assessment using condition number and VIF
* Residual diagnostics
* Breusch–Pagan test for heteroscedasticity

The OLS regression results showed that `Age`, `Annual Salary`, and `Net Worth` were statistically significant positive predictors of `Car Purchase Amount`, while `Gender` was not statistically significant.

### 4. Feature Preparation and Scaling

The target variable was defined as:

```text
Car Purchase Amount
```

The predictor variables were:

```text
Gender
Age
Annual Salary
Credit Card Debt
Net Worth
```

Feature scaling was applied using `StandardScaler` to improve numerical stability, especially for regression models using regularization. The dataset was split into training and testing sets using a fixed random state for reproducibility.

### 5. Model Development
#### Linear Regression with Non-Negative Least Squares

A constrained Linear Regression model was fitted using `positive=True`. This approach was used because car purchase amount is a non-negative financial quantity, and the model should avoid unrealistic negative coefficient behavior where appropriate.

### 6. Model Evaluation

Model performance was evaluated using:

* Mean Absolute Error
* Mean Squared Error
* Root Mean Squared Error
* R² Score
* Mean Absolute Percentage Error
* Actual vs Predicted plots
* Residual analysis

The Linear Regression model achieved very strong performance, with a test R² of approximately **0.9994**, MAE around **217.74**, and RMSE around **253.53**.

***

## Key Findings

### Strong Predictive Performance

The model predicted car purchase amount with very high accuracy. The predicted values were very close to the actual values, and the average prediction error was small compared with the average purchase amount.

From the prediction error summary:

* Mean actual value: approximately **45,119.14**
* Mean predicted value: approximately **45,088.76**
* Mean residual: approximately **30.37**
* Mean absolute error: approximately **217.12**
* Maximum absolute error: approximately **458.98**

The residual mean was close to zero, indicating that the model did not strongly overpredict or underpredict.

### Important Predictors

The statistical regression analysis indicated that:

* `Age` was a significant positive predictor.
* `Annual Salary` was a significant positive predictor.
* `Net Worth` was a significant positive predictor.
* `Gender` was not statistically significant.

This suggests that financial capacity and age-related factors are more important than gender for explaining car purchase amount.

### OLS Regression Insights

The OLS model explained a large proportion of variation in car purchase amount, with an R² value around **0.964**. The overall model was statistically significant based on the F-test. Age, Annual Salary, and Net Worth had statistically significant positive coefficients, while Gender did not contribute significantly.

### Model Diagnostics

Residual diagnostic tests were used to evaluate regression assumptions. The initial Breusch–Pagan test suggested heteroscedasticity. After applying a corrected/robust modeling approach, the Breusch–Pagan test became non-significant, indicating that the heteroscedasticity issue was substantially addressed.

### Practical Interpretation

The model can be used to estimate how much a customer may spend on a car based on financial and demographic information. Customers with higher annual salary, higher net worth, and older age tend to have higher predicted car purchase amounts.

***

## Final Conclusion

This project demonstrates a complete regression-based prediction pipeline for estimating car purchase amount. The final model achieved excellent predictive performance and was supported by inferential statistical validation.

The analysis shows that **Age**, **Annual Salary**, and **Net Worth** are the most important predictors of car purchase amount, while **Gender** does not appear to have a statistically significant effect. The model produced low prediction errors and strong agreement between actual and predicted values.

Although the results are highly accurate, the dataset appears clean and strongly linear. Therefore, future work should validate the model on more realistic customer data and compare performance with non-linear models.

***

## Future Improvements

Future extensions of this project may include:
* Testing non-linear models such as Random Forest Regressor and XGBoost Regressor
* Including additional customer behavior variables
* Building a simple Streamlit app for prediction
* Adding model explainability using SHAP or permutation importance
