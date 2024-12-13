# Linear Regression: A Quick Overview

Before you dive into the implementations, I highly recommend first learning the heart of each algorithm—its core idea and how it works. You can explore this through YouTube tutorials, books, or online courses. This repository is meant to complement that knowledge by showing how to translate concepts into working code.

Linear regression is a regression algorithm used to predict continuous values. It's one of the simplest and most interpretable models in machine learning, making it a great starting point for understanding regression concepts.

## Key Idea

Linear regression assumes a linear relationship between the features (x) and the target (y). It tries to model this relationship using the equation: 

**y = mx + b**

Where:

- **m**: The slope or weight, which determines the rate of change in y for a unit change in x.
- **b**: The intercept, which represents the value of y when x = 0.

The goal of the model is to find the best-fit line that minimizes the error between the predicted values (**ŷ**) and the actual values (**y**).

## How Does It Work?

### Finding the Best Fit Line

The model needs to determine the values of **m** (slope) and **b** (intercept) such that the line best represents the relationship (patterns) between the features and the target. This is done by minimizing the error between the predicted and actual values.

### Gradient Descent

The optimization process uses **gradient descent** to iteratively update the values of **m** and **b**. 

- Gradient descent works by calculating the gradient (or slope of the error curve) with respect to the parameters and moving in the direction that reduces the error.
- The **learning rate** determines how large each step is during the optimization process. A small learning rate makes convergence slow, while a large one might overshoot the optimal solution.

### Cost Function

To quantify the error, we use a cost function, such as **Mean Squared Error (MSE)**:

\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
\]

Where:
- **n**: Number of data points.
- **ŷᵢ**: Predicted value for the i-th data point.
- **yᵢ**: Actual value for the i-th data point.

The goal is to minimize this cost function during training. A smaller error indicates a better fit.

## Evaluation Metrics

After training the model, we evaluate its performance using metrics such as:

- **Mean Squared Error (MSE)**: Measures the average squared error between predictions and actual values. Smaller is better.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, making the error interpretable in the same units as the target.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predictions and actual values. It is less sensitive to outliers than MSE.
- **R² (Coefficient of Determination)**: Explains the proportion of variance in the target that is predictable from the features. Values close to 1 indicate a good fit.

## Key Points to Note

### Assumptions of Linear Regression
- There is a linear relationship between the features and the target.
- Errors are normally distributed and independent.
- **Homoscedasticity**: The variance of errors is constant across predictions.
- Features are not highly correlated with each other (no multicollinearity).

### When to Use Linear Regression
- Works well for smaller datasets with linear relationships.
- Ideal when interpretability is crucial (e.g., understanding feature importance through weights).

### Limitations
- Struggles with non-linear relationships.
- Sensitive to outliers, as they can significantly affect the slope (**m**).
- Assumes all relevant variables are included and measured accurately.

### Why Is Linear Regression Popular?
- It's simple and easy to interpret.
- Training is computationally efficient.
- It serves as a baseline for comparing more complex regression models, including neural networks.

---

# Linear Regression Implementation

This project demonstrates the implementation of a simple linear regression model using Python, along with libraries such as numpy, pandas, matplotlib, and scikit-learn. The example is designed to predict continuous values based on a single feature and target dataset.

### Dataset

The dataset consists of:
- **Features**: Independent variable(s) (e.g., [1, 2, 3, ...]).
- **Targets**: Dependent variable(s) to predict (e.g., [10, 25, 30, ...]).

We load this dataset using pandas.

## Steps in the Code

### Import Libraries

Libraries like numpy and pandas are used for data handling. matplotlib is used for visualizations. scikit-learn is used for modeling and evaluation.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

Load Dataset

The dataset is read from a CSV file using pandas. Features and targets are extracted to create a DataFrame. The first few rows are displayed using print.
data = pd.read_csv('dataset.csv')
features = data['features']
targets = data['targets']
df = pd.DataFrame({'features': features, 'targets': targets})
print(df.head(5))

Visualize Data

A scatter plot is created to show the relationship between features and targets. This step helps us understand the data before applying the model.
plt.scatter(df['features'], df['targets'], c='red')
plt.xlabel('features')
plt.ylabel('targets')
plt.title('features vs targets')
plt.show()

Fit the Linear Regression Model

A LinearRegression object is created and trained using the features and targets. The model learns the best-fit line parameters (slope and intercept).
model = LinearRegression()
model.fit(df[['features']], df[['targets']])

Make Predictions

The model predicts target values for the given features. These predictions are stored for visualization and evaluation.
predictions = model.predict(df[['features']])

Visualize the Best Fit Line

The best-fit line is plotted over the scatter plot of the data points. This visualization helps confirm that the model has captured the trend in the data.
plt.scatter(df['features'], df['targets'], c='red')
plt.plot(df['features'], predictions, c='green', label='Best Fit Line')
plt.legend()
plt.show()

Evaluate the Model

Two metrics are used for evaluation:

    Mean Squared Error (MSE): Captures the average squared difference between actual and predicted values.
    Mean Absolute Error (MAE): Measures the average absolute difference between actual and predicted values.

mse = mean_squared_error(df['targets'], predictions)
print('Mean Squared Error:', mse)

mae = mean_absolute_error(df['targets'], predictions)
print('Mean Absolute Error:', mae)

Test the Model

The model is tested by making predictions for unseen feature values (e.g., 25). The predicted target is printed.
print(model.predict([[25]]))
[[243.04761905]]
Key Takeaways

    The implementation uses a simple dataset for clarity.
    The model captures the linear trend in the data effectively, as seen in the best-fit line.
    Evaluation metrics like MSE and MAE provide insight into the model's error.

This example serves as a foundation for applying linear regression to more complex datasets.
