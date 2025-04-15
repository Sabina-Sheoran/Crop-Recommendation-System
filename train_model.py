import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("C:\ml2 project\Crop_recommendation.csv")

# Features & labels
X = df.drop(columns=["label"])
y = df["label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as 'crop_model.pkl'")
