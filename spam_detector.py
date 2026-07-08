import pandas as pd

df = pd.read_csv("spam_data.csv")

print(df.head())
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["Message"])
y = df["Spam"]

model = MultinomialNB()
model.fit(X, y)

test_message = ["Congratulations! You won a free laptop"]

test_data = vectorizer.transform(test_message)

prediction = model.predict(test_data)

if prediction[0] == 1:
    print("Prediction: Spam Message")
else:
    print("Prediction: Not Spam")
test_message2 = ["Meeting is scheduled at 10 AM"]

test_data2 = vectorizer.transform(test_message2)

prediction2 = model.predict(test_data2)

if prediction2[0] == 1:
    print("Prediction: Spam Message")
else:
    print("Prediction: Not Spam")
