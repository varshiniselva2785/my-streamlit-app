import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load merged data
data = pd.read_csv("data/final_dengue_data.csv")

# Features and target
X = data[["temperature", "humidity", "rainfall"]]
y = data["dengue_cases"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "dengue_model.pkl")

print("✅ Model trained successfully")
print("📁 Model saved as dengue_model.pkl")

