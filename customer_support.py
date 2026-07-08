import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("tickets.csv")

X = data["ticket"]
y = data["category"]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vector, y)

# User input
user_ticket = input("Enter your support issue: ")

# Predict category
user_vector = vectorizer.transform([user_ticket])
prediction = model.predict(user_vector)

print("\nPredicted Category:", prediction[0])
