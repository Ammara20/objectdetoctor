import cv2
import numpy as np
import tensorflow.keras as keras

# Load the class names
class_names = []
with open('coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Load the pre-trained model
model = keras.models.load_model('path/to/model')

# Initialize the video stream
with cv2.VideoCapture(0) as cap:
    # Check if the video stream was opened successfully
    if not cap.isOpened():
        print("Error opening video stream")
        exit()

    while True:
        # Capture a frame
        ret, frame = cap.read()

        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Perform object detection
        predictions = model.predict(frame)
        class_id = np.argmax(predictions)
        class_name = class_names[class_id]

        # Draw bounding boxes around the objects
        if class_id > 0:
            bbox = predictions[0][class_id][1:]
            (x, y, w, h) = bbox * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow('Objects Detected', frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()