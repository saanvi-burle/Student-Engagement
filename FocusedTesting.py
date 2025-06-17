# test_model.py
import tensorflow as tf
import numpy as np
import cv2

# Load the saved model
model = tf.keras.models.load_model('focus_detection_transfer_model.h5')

# Parameters
img_width, img_height = 224, 224
labels = {0: 'Not Focused', 1: 'Focused'}

def process_frame(frame):
    img = cv2.resize(frame, (img_width, img_height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict class
    prediction = model.predict(img)
    label = labels[int(prediction[0] > 0.5)]
    
    # Draw rectangle and label
    cv2.rectangle(frame, (50, 50), (300, 100), (0, 255, 0), 2)
    cv2.putText(frame, label, (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def detect_focus(source=0):
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame for detection
        frame = process_frame(frame)

        # Show the result
        cv2.imshow('Focus Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Test with live camera
detect_focus(0)

# Uncomment to test with a video file
# detect_focus('path_to_video.mp4')
