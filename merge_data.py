import pandas as pd

dengue = pd.read_csv("data/dengue_cases.csv")
weather = pd.read_csv("data/weather_data.csv")

merged = pd.merge(dengue, weather, on="date")

merged.to_csv("data/final_dengue_data.csv", index=False)

print("Merged data created successfully")
print(merged)


