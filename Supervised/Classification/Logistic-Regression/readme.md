<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logistic Regression Explanation</title>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
    <h1>Full Explanation of Logistic Regression</h1>
    <p>Logistic Regression is a popular algorithm used for binary classification tasks. It is a type of generalized linear model (GLM) where the output is a probability value between 0 and 1, which is mapped to a binary outcome (class 0 or class 1). Despite its name, logistic regression is used for classification, not regression. Below is a detailed explanation of logistic regression.</p>

    <h2>1. Binary Classification Problem</h2>
    <p>In binary classification, we are tasked with classifying data points into one of two classes, typically represented as <code>0</code> and <code>1</code>. For example:</p>
    <ul>
        <li>Class 0: "Not an email"</li>
        <li>Class 1: "Spam email"</li>
    </ul>
    <p>The goal is to find the best decision boundary (or plane in higher dimensions) that separates the two classes.</p>

    <h2>2. Model Setup</h2>
    <p>In logistic regression, the model uses the following equation to predict the probability of an input data point <code>x</code> belonging to class 1:</p>
    <p>
        \[
        P(y=1 | x) = \sigma(w^T x + b)
        \]
    </p>
    <p>Where:</p>
    <ul>
        <li><code>w</code> is a vector of weights (or parameters) for each feature in the input vector <code>x</code>,</li>
        <li><code>x</code> is the input feature vector (with a bias term added),</li>
        <li><code>b</code> is the bias term (the intercept),</li>
        <li><code>\sigma(z) = \frac{1}{1 + e^{-z}}</code> is the sigmoid function which squashes the linear combination of features into a probability value between 0 and 1.</li>
    </ul>
    <p>The sigmoid function, <code>\sigma(z)</code>, has the following behavior:</p>
    <ul>
        <li>When <code>z \to \infty</code>, <code>\sigma(z) \to 1</code> (indicating class 1).</li>
        <li>When <code>z \to -\infty</code>, <code>\sigma(z) \to 0</code> (indicating class 0).</li>
    </ul>
    <p>The value <code>P(y=1 | x)</code> represents the probability that the model assigns the label 1 to the input <code>x</code>. To predict class labels, we use a threshold of 0.5:</p>
    <ul>
        <li>If <code>P(y=1 | x) \geq 0.5</code>, classify as class 1.</li>
        <li>If <code>P(y=1 | x) < 0.5</code>, classify as class 0.</li>
    </ul>

    <h2>3. Cost Function (Log-Loss)</h2>
    <p>To train a logistic regression model, we need to find the optimal parameters (weights <code>w</code> and bias <code>b</code>) that best fit the data. We do this by minimizing a loss function (also called the cost function) that quantifies how well the model's predictions match the true labels.</p>
    <p>The log loss (also called binary cross-entropy) is used as the loss function in logistic regression. It calculates the error between the predicted probability and the true label for each data point.</p>
    <p>For a single data point, the log loss is:</p>
    <p>
        \[
        \text{log loss}(y, p) = -[y \log(p) + (1 - y) \log(1 - p)]
        \]
    </p>
    <p>Where:</p>
    <ul>
        <li><code>y</code> is the true label (0 or 1),</li>
        <li><code>p</code> is the predicted probability (output of the sigmoid function).</li>
    </ul>
    <p>For the entire dataset with <code>N</code> data points, the average log loss (or cost) is:</p>
    <p>
        \[
        J(w, b) = \frac{1}{N} \sum_{i=1}^{N} \left[ -y_i \log(\hat{y}_i) - (1 - y_i) \log(1 - \hat{y}_i) \right]
        \]
    </p>
    <p>Where:</p>
    <ul>
        <li><code>\hat{y}_i</code> is the predicted probability for the <code>i</code>-th data point, i.e., <code>\hat{y}_i = \sigma(w^T x_i + b)</code>,</li>
        <li><code>y_i</code> is the true label for the <code>i</code>-th data point.</li>
    </ul>

    <h2>4. Optimization and Gradient Descent</h2>
    <p>To find the optimal parameters (weights <code>w</code> and bias <code>b</code>) that minimize the log loss, we use gradient descent or another optimization technique. The gradient of the cost function with respect to the parameters is computed and used to update the weights and bias.</p>
    <p>The gradients of the cost function with respect to the weights <code>w</code> and bias <code>b</code> are computed using calculus. For the weights, the gradient is:</p>
    <p>
        \[
        \frac{\partial J(w, b)}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) x_{ij}
        \]
    </p>
    <p>Where <code>x_{ij}</code> is the <code>j</code>-th feature of the <code>i</code>-th data point.</p>
    <p>For the bias term, the gradient is:</p>
    <p>
        \[
        \frac{\partial J(w, b)}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
        \]
    </p>
    <p>The gradient descent algorithm updates the weights and bias in the opposite direction of the gradient to minimize the cost function:</p>
    <p>
        \[
        w_j := w_j - \alpha \frac{\partial J(w, b)}{\partial w_j}
        \]
    </p>
    <p>
        \[
        b := b - \alpha \frac{\partial J(w, b)}{\partial b}
        \]
    </p>
    <p>Where:</p>
    <ul>
        <li><code>\alpha</code> is the learning rate, a small positive value that controls the step size.</li>
    </ul>
    <p>The parameters are iteratively updated until the cost function converges to a minimum.</p>

    <h2>5. Prediction</h2>
    <p>Once the model has been trained (i.e., the weights <code>w</code> and bias <code>b</code> have been optimized), we can use it to make predictions on new, unseen data. Given a new input <code>x</code>, we compute the predicted probability:</p>
    <p>
        \[
        P(y=1 | x) = \sigma(w^T x + b)
        \]
    </p>
    <p>Then, based on the threshold (typically 0.5), we classify it as class 1 or class 0.</p>
</body>
</html>


