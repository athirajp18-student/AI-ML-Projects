import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("news.csv")

# Features and Labels
X = data["text"]
y = data["label"]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vector, y)

# Test news
test_news = ["Aliens are living inside the Earth"]

test_vector = vectorizer.transform(test_news)

prediction = model.predict(test_vector)

print("Prediction:", prediction[0])
