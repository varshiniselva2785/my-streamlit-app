import pandas as pd
import numpy as np

# Load files
weather = pd.read_csv("data/weather_data.csv")
dengue = pd.read_csv("data/dengue_data.csv")

# Merge
final_data = pd.merge(weather, dengue, on="date")

# Add coordinates (important for GIS)
final_data["latitude"] = np.random.uniform(10.00, 10.35, len(final_data))
final_data["longitude"] = np.random.uniform(77.40, 77.65, len(final_data))

# Save
final_data.to_csv("data/final_dengue_data.csv", index=False)

print("✅ final_dengue_data.csv created successfully!")
