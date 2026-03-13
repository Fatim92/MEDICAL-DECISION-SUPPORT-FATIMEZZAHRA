import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Charger les données
data = pd.read_csv("Data/ObesityDataSet_raw_and_data_sinthetic.csv")

# Séparer X et y
X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

# Transformer les variables catégorielles
X = pd.get_dummies(X)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, "src/model.pkl")

print("✅ Model trained successfully")
print("📁 Model saved as src/model.pkl")