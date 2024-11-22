from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from data_generation import weights
import logging

app = Flask(__name__)

loaded_model = joblib.load("models/model_6.pkl")

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
    (speed_limit, speeding_event, braking_event, accelerating_event, cornering_event, distracted_camevent, 
    following_distance_camevent, trip_distance) = user_data

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate_driver', methods=['POST', 'GET'])
def evaluate_driver():
    try:

        logging.debug(f"Form data recieved: {request.form}")

        # Check if all required fields are present
        required_fields = ['speed_limit', 'speeding_event', 'braking_event', 'accelerating_event', 
                           'cornering_event', 'distracted_camevent', 'following_distance_camevent', 'driver_id', 'trip_distance']
        for field in required_fields:
            if field not in request.form:
                raise ValueError(f"Missing required field: {field}")

        # Get form data
        speed_limit = int(request.form['speed_limit'])
        speeding_event = int(request.form['speeding_event'])
        braking_event = int(request.form.get('braking_event'), 0)
        accelerating_event = int(request.form.get('accelerating_event'), 0)
        cornering_event = int(request.form['cornering_event'])
        distracted_camevent = int(request.form.get('distracted_camevent'), 0)
        following_distance_camevent = int(request.form.get('following_distance_camevent'), 0)
        trip_distance = int(request.form['trip_distance'])

        # Format input for the model
        user_data = np.array([[speed_limit, speeding_event, braking_event,
                                accelerating_event, cornering_event, distracted_camevent,
                                following_distance_camevent, trip_distance]])
        
        # Get user ID
        driver_id = request.form['driver_id']

        # Make a prediction
        predicted_score = loaded_model.predict(user_data)[0]

        # Generate recommendations
        recommendations = provide_recommendations(user_data[0])

        results = {
            "Driver_ID": driver_id,
            "Safety_Score": round(predicted_score, 2),
            "Recommendations": recommendations
        }
        return jsonify(results)

    except Exception as e:
        logging.error(f"An error occured: {str(e)}")
        return(jsonify({'error': str(e)})), 400

if __name__ == '__main__':
    app.run(debug=True)