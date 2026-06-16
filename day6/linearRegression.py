# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
# Let's say a student studies between 1 and 10 hours
X = np.random.uniform(1, 10, 100).reshape(-1, 1)  # 100 samples, 1 feature

# Generate scores with some noise: score = 5 * hours + noise
noise = np.random.normal(0, 3, 100).reshape(-1, 1)
y = 5 * X + noise  # linear relationship + noise

# Convert to DataFrame for visualization
df = pd.DataFrame({'Study Hours': X.flatten(), 'Score': y.flatten()})
print(df.head())

# Scatter plot of the generated data
plt.figure(figsize=(8, 5))
plt.scatter(df['Study Hours'], df['Score'], color='blue')
plt.title("Study Hours vs Student Score")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.grid(True)
plt.show()


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🔍 Model Evaluation Metrics")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot the best fit line on training data
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')
plt.title("Linear Regression Fit on Training Data")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()


# Plot the residual errors
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='green', label="Actual")
plt.scatter(X_test, y_pred, color='red', label="Predicted")
for i in range(len(X_test)):
    plt.plot([X_test[i], X_test[i]], [y_test[i], y_pred[i]], color='gray', linestyle='--')

plt.title("Prediction vs Actual with Error Lines")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

#find the model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy (R² Score): {accuracy:.2f}")