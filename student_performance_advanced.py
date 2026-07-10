import pandas as pd

df = pd.read_csv("student data.csv")

print(df.head())
from sklearn.linear_model import LinearRegression

X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

new_student = pd.DataFrame({
    "Study_Hours": [10],
    "Attendance": [95]
})

prediction = model.predict(new_student)

print("Predicted Marks:", prediction[0])
import matplotlib.pyplot as plt

plt.scatter(df["Study_Hours"], df["Marks"], color="blue")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = df[["Study_Hours", "Attendance"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))

