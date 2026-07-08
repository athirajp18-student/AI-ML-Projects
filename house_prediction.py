import pandas as pd

df = pd.read_csv("house_data.csv")

print(df.head())
from sklearn.linear_model import LinearRegression

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

new_house = pd.DataFrame({
    "Area": [1900],
    "Bedrooms": [3],
    "Age": [2]
})

prediction = model.predict(new_house)

print("Predicted House Price: ₹", prediction[0])
import matplotlib.pyplot as plt

plt.scatter(df["Area"], df["Price"])
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price")
plt.title("House Price Prediction")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
