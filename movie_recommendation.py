import pandas as pd

df = pd.read_csv("movies.csv")

print(df)
movie = input("Enter a genre (Action/Romance/Sci-Fi): ")

recommendations = df[df["Genre"].str.lower() == movie.lower()]

print("\nRecommended Movies:")
print(recommendations["Movie"])
