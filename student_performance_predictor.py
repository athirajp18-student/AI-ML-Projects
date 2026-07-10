import pandas as pd

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 85, 90, 95]
}

df = pd.DataFrame(data)

print(df)
from sklearn.linear_model import LinearRegression

X = df[["Study_Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[12]])

print("Predicted Marks for 12 hours of study:", prediction[0])
import matplotlib.pyplot as plt

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Performance Prediction")
plt.legend()

plt.show()
