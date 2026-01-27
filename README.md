# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Deepak K R
RegisterNumber:  25008695 (212224050057)
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Data
# -----------------------
df = pd.read_csv('ex3.csv')
x_raw = df['R&D Spend'].values
y_raw = df['Profit'].values

# Feature Scaling (Min-Max) to ensure convergence with Gradient Descent
x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

# -----------------------
# Parameters
# -----------------------
w = 0.0
b = 0.0
alpha = 0.1
epochs = 100
n = len(x)

losses = []

# -----------------------
# Gradient Descent
# -----------------------
for _ in range(epochs):
    y_hat = w * x + b

    # Mean Squared Error
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

# -----------------------
# Plots
# -----------------------
plt.figure(figsize=(12, 5))

# 1️⃣ Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses, color="blue")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("R&D Spend VS Profit")

# 2️⃣ Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, color="red", label="Data")
plt.plot(x, w * x + b, color="green", label="Regression Line")
plt.xlabel("R&D Spend (Scaled)")
plt.ylabel("Profit (Scaled)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)

```

## Output:
![IMG](<Screenshot 2026-01-27 114531.png>)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
