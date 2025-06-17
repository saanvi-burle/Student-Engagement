from flask import Flask, render_template, request, jsonify, url_for
import cv2
import numpy as np
from keras.models import model_from_json
import psycopg2
import plotly.graph_objs as go
import json

app = Flask(__name__, static_folder='static')

# Updated labels with corresponding numeric values
emotion_dict = {0: "Distracted", 1: "Attentive", 2: "Neutral", 3: "Less Distracted", 4: "Less Attentive"}
emotion_numeric = {"Attentive": 5, "Less Attentive": 4, "Neutral": 3, "Less Distracted": 2, "Distracted": 1}

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to run the model and store data in PostgreSQL 
@app.route('/run-model', methods=['POST'])
def run_model():
    # Load model
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model.weights.h5")
    print("Loaded model from disk")

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="emotion_detection",
        user="postgres",
        password="saanvi123",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # Start video feed
    cap = cv2.VideoCapture("zoom.mp4") #replace with your video if needed


    #if capturing from camera, uncomment below statement and comment above statement

    # cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            detected_emotion = emotion_dict[maxindex]

            # Insert the data into the PostgreSQL database
            insert_query = """
            INSERT INTO emotion_data (timestamp, emotion) 
            VALUES (%s, %s)
            """
            cur.execute(insert_query, (timestamp, detected_emotion))
            
            # Commit to save the insertion
            conn.commit()

            print(f"Inserting: Time - {timestamp}, Detected Emotion - {detected_emotion}")

            # Display the emotion on the frame
            cv2.putText(frame, detected_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Close the database connection
    cur.close()
    conn.close()

    return "Model run successfully!", 200

# Route to fetch data from the database and create interactive graphs
@app.route('/plot-graph', methods=['GET'])
def plot_graph():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="emotion_detection",
        user="postgres",
        password="saanvi123",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # Query to fetch timestamp and emotion
    fetch_query = "SELECT timestamp, emotion FROM emotion_data ORDER BY timestamp ASC"
    cur.execute(fetch_query)
    data = cur.fetchall()

    # Close the connection
    cur.close()
    conn.close()

    # Process the data
    timestamps = [row[0] for row in data]
    emotions = [row[1] for row in data]

    print("Fetched data:", list(zip(timestamps, emotions)))  # Debugging

    # Create line graph data
    numeric_emotions = [emotion_numeric[emotion] for emotion in emotions]

    # Count occurrences of each emotion category
    categories = ['Attentive', 'Less Attentive', 'Neutral', 'Less Distracted', 'Distracted']
    counts = {cat: 0 for cat in categories}
    
    for emotion in emotions:
        if emotion == "Attentive":
            counts['Attentive'] += 1
        elif emotion == "Less Attentive":
            counts['Less Attentive'] += 1
        elif emotion == "Neutral":
            counts['Neutral'] += 1
        elif emotion == "Less Distracted":
            counts['Less Distracted'] += 1
        elif emotion == "Distracted":
            counts['Distracted'] += 1

    # Prepare data for plotly
    bar_x = list(counts.keys())
    bar_y = list(counts.values())

    # Calculate average focused and unfocused
    total_attentive = counts['Attentive'] + counts['Less Attentive']
    total_unfocused = counts['Distracted'] + counts['Less Distracted']
    neutral = counts['Neutral']

    avg_focused = total_attentive / len(emotions) * 100 if len(emotions) > 0 else 0
    avg_unfocused = total_unfocused / len(emotions) * 100 if len(emotions) > 0 else 0
    avg_neutral = neutral / len(emotions) * 100 if len(emotions) > 0 else 0

    # Return JSON response
    return jsonify({
        "line_data": {
            "timestamps": timestamps,
            "emotions": numeric_emotions,
        },
        "bar_data": {
            "bar_x": bar_x,
            "bar_y": bar_y,
            "avg_focused": avg_focused,
            "avg_unfocused": avg_unfocused,
            "avg_neutral": avg_neutral,
        }
    })

if __name__ == '__main__':
    app.run(debug=True)


# cap = cv2.VideoCapture("C:/Users/saanvisburle/Desktop/MIT-Saanvi/Engg/TY/Project/Emotion_detection_with_CNN-main/dance.mp4")