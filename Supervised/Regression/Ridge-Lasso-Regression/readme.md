Ridge and Lasso Regression: A Detailed Explanation

Understanding Ridge and Lasso Regression

Linear regression is a fundamental algorithm used to predict continuous values by finding the best-fit line for the given data. However, it can suffer from overfitting when the dataset has many features or multicollinearity among them. Ridge and Lasso regression address these issues by introducing regularization, which penalizes large coefficient values, helping to improve model generalization.

Linear Regression:

Objective: Minimize the Residual Sum of Squares (RSS).

Drawback: No penalty for large coefficients, which can lead to overfitting in high-dimensional data.

Ridge Regression:

Objective: Minimize RSS with an additional penalty term:

\text{Objective: } \text{Minimize } \text{RSS} + \lambda \sum_{i=1}^{n} \beta_i^2

where:

 (alpha in scikit-learn) is the regularization parameter.

 are the model coefficients.

Effect: Shrinks coefficients towards zero but never eliminates them. Useful when all features are important but need regularization.

Key Point: Ridge does not perform feature selection; it retains all features with reduced impact.

Lasso Regression:

Objective: Minimize RSS with an additional penalty term:

\text{Objective: } \text{Minimize } \text{RSS} + \lambda \sum_{i=1}^{n} |\beta_i|

where:

 is the regularization parameter.

 is the absolute value of the coefficients.

Effect: Shrinks some coefficients to exactly zero, effectively eliminating less important features.

Key Point: Lasso is useful for feature selection in addition to regularization.

Comparing Ridge, Lasso, and Linear Regression:

Feature

Linear Regression

Ridge Regression

Lasso Regression

Overfitting

High

Reduced

Reduced

Feature Selection

No

No

Yes

Penalty

None

 (L2 Norm)

(

\beta

) (L1 Norm)

Coefficients

Large

Shrinks, no elimination

Shrinks, can eliminate

Code Explanation

Dataset and Libraries:

The code uses the Diabetes Dataset from sklearn, a small dataset often used for regression problems. Libraries like pandas, numpy, and matplotlib are used for data manipulation and visualization.

from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Dataset Preparation:

The dataset is loaded and split into training and test sets. A DataFrame is created for easy viewing and manipulation.

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
df = pd.DataFrame(X, columns=diabetes.feature_names)
df['target'] = y

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Linear Regression:

A simple linear regression model is trained and evaluated. The R-squared score and Mean Squared Error (MSE) are computed for the test set.

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_score = linear_model.score(X_test, y_test)
linear_mse = mean_squared_error(y_test, linear_model.predict(X_test))

Ridge Regression ():

This is equivalent to ordinary least squares (OLS) regression as the penalty term is disabled. The model is trained and evaluated.

ridge_model_ols = Ridge(alpha=0)
ridge_model_ols.fit(X_train, y_train)
ridge_ols_score = ridge_model_ols.score(X_test, y_test)
ridge_ols_mse = mean_squared_error(y_test, ridge_model_ols.predict(X_test))

Ridge Regression with Cross-Validation:

The RidgeCV class finds the best  using cross-validation over a range of values. The best  is then used to train and evaluate the final model.

ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5)
ridge_cv.fit(X_train, y_train)
best_alpha_ridge = ridge_cv.alpha_

ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train, y_train)
ridge_score = ridge_model.score(X_test, y_test)
ridge_mse = mean_squared_error(y_test, ridge_model.predict(X_test))

Lasso Regression ():

The Lasso regression with  is equivalent to OLS. However, it is generally not recommended to set  for Lasso due to computational inefficiency.

lasso_alpha_0 = Lasso(alpha=0, max_iter=10000)
lasso_alpha_0.fit(X_train, y_train)
lasso_alpha_0_score = lasso_alpha_0.score(X_test, y_test)
lasso_alpha_0_mse = mean_squared_error(y_test, lasso_alpha_0.predict(X_test))

Lasso Regression with Cross-Validation:

The LassoCV class finds the best  using cross-validation over a range of values. The best  is then used to train and evaluate the final model.

lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 7), cv=5)
lasso_cv.fit(X_train, y_train)
best_alpha_lasso = lasso_cv.alpha_

lasso_model = Lasso(alpha=best_alpha_lasso)
lasso_model.fit(X_train, y_train)
lasso_score = lasso_model.score(X_test, y_test)
lasso_mse = mean_squared_error(y_test, lasso_model.predict(X_test))

Visualizations:

Coefficients Plot:

Compares coefficients across Linear, Ridge, and Lasso models.

Demonstrates the impact of regularization.

models = ['Linear', 'Ridge (Best Alpha)', 'Lasso (Best Alpha)']
coefficients = [
    linear_model.coef_,
    ridge_model.coef_,
    lasso_model.coef_
]

plt.figure(figsize=(10, 6))
for i, coef in enumerate(coefficients):
    plt.plot(coef, label=models[i])

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(ticks=range(len(diabetes.feature_names)), labels=diabetes.feature_names, rotation=45)
plt.title('Model Coefficients')
plt.ylabel('Coefficient Value')
plt.legend()
plt.tight_layout()
plt.show()

R-squared Scores Bar Chart:

Compares R-squared scores across the three models.

scores = [linear_score, ridge_score, lasso_score]
plt.figure(figsize=(8, 5))
plt.bar(models, scores, color=['blue', 'green', 'red'])
plt.title('Model R-squared Scores')
plt.ylabel('R-squared')
plt.ylim(0, 1)
plt.show()

Key Takeaways:

Linear Regression is simple but prone to overfitting in high-dimensional data.

Ridge Regression reduces overfitting but retains all features with smaller coefficients.

Lasso Regression reduces overfitting and performs feature selection by eliminating irrelevant features.

Visualizations help understand the impact of regularization on model performance and coefficients.
