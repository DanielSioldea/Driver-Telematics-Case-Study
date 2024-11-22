from data_generation import df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Features for Linear Regression model
features = [
    "Speed_Limit",
    "Speeding_Event",
    "Braking_Event",
    "Accelerating_Event",
    "Cornering_Event",
    "Distracted_CamEvent",
    "Following_Distance_CamEvent",
    "Trip_Distance"
]

# Target variable for model - Calculated Safety Score
target = "Safety_Score"

# Convert boolean columns to integers - to deal with checkboxes in the form
df[features[2:]] = df[features[2:]].astype(int)

# Splitting the data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Saving the model
joblib.dump(model, "models/model_6.pkl")