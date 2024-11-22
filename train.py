import joblib
import numpy as np
import warnings
from data_generation import weights
warnings.filterwarnings("ignore")

model = joblib.load("models/model3.pkl")

features = [
    "Speed_Limit",
    "Speeding_Event",
    "Braking_Event",
    "Accelerating_Event",
    "Cornering_Event",
    "Distracted_CamEvent",
    "Following_Distance_CamEvent"
]

# Event to Recommendation Mapping
event_recommendations = {
    "Speeding_Event": "reduce your speed to stay within the speed limit",
    "Braking_Event": "avoid harsh braking to maintain smoother driving",
    "Accelerating_Event": "reduce sudden acceleration to improve safety",
    "Cornering_Event": "take corners more gently to avoid risks",
    "Distracted_CamEvent": "stay focused and avoid distractions while driving",
    "Following_Distance_CamEvent": "maintain a safe following distance to prevent collisions"
}

# Function to determine the top events and provide recommendations
def provide_recommendations(user_data):
    # Extract user inputs
    speed_limit = user_data[0][0]
    speeding_event = user_data[0][1]
    braking_event = user_data[0][2]
    accelerating_event = user_data[0][3]
    cornering_event = user_data[0][4]
    distracted_camevent = user_data[0][5]
    following_distance_camevent = user_data[0][6]

    # Check which events occurred
    events = {
        "Speeding_Event": speeding_event > speed_limit,
        "Braking_Event": braking_event,
        "Accelerating_Event": accelerating_event,
        "Cornering_Event": cornering_event,
        "Distracted_CamEvent": distracted_camevent,
        "Following_Distance_CamEvent": following_distance_camevent
    }

    # Rank events by weights (if occurred)
    occurred_events = {event: weights[event] for event, occurred in events.items() if occurred}
    top_events = sorted(occurred_events.items(), key=lambda x: x[1], reverse=True)[:2]

    # Generate recommendations
    recommendations = [event_recommendations[event[0]] for event in top_events]
    return " and ".join(recommendations).capitalize() + " to improve your safety score." if recommendations else "Keep up the safe driving!"

def get_user_prediction():
    print("Enter your driving data:")
    
    # Collect user input
    try:
        speed_limit = int(input("Speed Limit (km/h): "))
        speeding_event = int(input("Driver Speed (km/h): "))
        braking_event = int(input("Braking Event (1 for Yes, 0 for No): "))
        accelerating_event = int(input("Accelerating Event (1 for Yes, 0 for No): "))
        cornering_event = int(input("Cornering Event (1 for Yes, 0 for No): "))
        distracted_camevent = int(input("Distracted Cam Event (1 for Yes, 0 for No): "))
        following_distance_camevent = int(input("Following Distance Cam Event (1 for Yes, 0 for No): "))
    except ValueError:
        print("Invalid input. Please enter numeric values where required.")
        return

    # Format the input into a numpy array
    user_data = np.array([[speed_limit, speeding_event, braking_event, 
                           accelerating_event, cornering_event, distracted_camevent, 
                           following_distance_camevent]])
    
    # Make a prediction
    predicted_score = model.predict(user_data)[0]
    print(f"\nPredicted Safety Score: {predicted_score:.2f}")

    # Provide recommendations
    recommendations = provide_recommendations(user_data)
    print(f"Recommendations: {recommendations}")

get_user_prediction()