import cv2
import numpy as np
import psycopg2
from keras.models import model_from_json
import time

# Emotion labels
emotion_dict = {0: "Disgusted", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

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
cap = cv2.VideoCapture("C:/Users/saanvisburle/Desktop/MIT-Saanvi/Engg/TY/Project/Emotion_detection_with_CNN-main/dance.mp4")

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

        print(f"Time: {timestamp:.2f} sec, Detected Emotion: {detected_emotion}")

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
