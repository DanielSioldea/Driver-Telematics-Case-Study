import pandas as pd
import numpy as np

# Create Simulated Dataset
n_drivers = 500
data = {
    "Driver_ID": [f"Driver_{i}" for i in range(1, n_drivers + 1)],
    "Timestamp": pd.date_range(start="2024-01-01", periods=n_drivers, freq="H"),
    "Speed_Limit": np.random.choice([50, 60, 80, 100], n_drivers),  # in km/h
    "Speeding_Event": np.random.randint(10, 150, n_drivers),  # in km/h
    "Braking_Event": np.random.choice([True, False], n_drivers, p=[0.4, 0.6]),
    "Accelerating_Event": np.random.choice([True, False], n_drivers, p=[0.2, 0.8]),
    "Cornering_Event": np.random.choice([True, False], n_drivers, p=[0.15, 0.85]),
    "Distracted_CamEvent": np.random.choice([True, False], n_drivers, p=[0.1, 0.9]),
    "Following_Distance_CamEvent": np.random.choice([True, False], n_drivers, p=[0.25, 0.75]),
    "Trip_Duration": np.random.randint(15, 120, n_drivers),  # in minutes
    "Trip_Distance": np.random.randint(5, 100, n_drivers),  # in km
    "Idle_Duration": np.random.randint(0, 30, n_drivers)  # in minutes
}

df = pd.DataFrame(data)

# Weights for each event to determine severity
weights = {
    "Speeding_Event": 0.1,
    "Braking_Event": 0.1,
    "Accelerating_Event": 0.2,
    "Cornering_Event": 0.15,
    "Distracted_CamEvent": 0.3,
    "Following_Distance_CamEvent": 0.15
}

# Safety Score Calculation
def calculate_safety_score(row):
    score = 100

    # Trip distance scaling factor
    if row["Trip_Distance"] < 10:
        dist_factor = 1.5  # Short trips get higher penalty
    elif row["Trip_Distance"] < 30:
        dist_factor = 1.2
    else:
        dist_factor = 1.0  # Long trips get normal penalty

    # Applying penalties for events
    if row["Speeding_Event"] > row["Speed_Limit"]:
        score -= weights["Speeding_Event"] * 100 * dist_factor

    if row["Braking_Event"]:
        score -= weights["Braking_Event"] * 100 * dist_factor
    
    if row["Accelerating_Event"]:
        score -= weights["Accelerating_Event"] * 100 * dist_factor

    if row["Cornering_Event"]:
        score -= weights["Cornering_Event"] * 100 * dist_factor

    if row["Distracted_CamEvent"]:
        score -= weights["Distracted_CamEvent"] * 100 * dist_factor

    if row["Following_Distance_CamEvent"]:
        score -= weights["Following_Distance_CamEvent"] * 100 * dist_factor

    return round(max(score, 0), 2)

df["Safety_Score"] = df.apply(calculate_safety_score, axis=1)

#print(df.head(20))
# print(df.to_string())
# print(df.shape)