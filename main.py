import cv2
import numpy as np

# Load the pre-trained model and configuration files
model_config = "yolov3.cfg"
model_weights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Load the class labels
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Access the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera (usually webcam)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame to match YOLO input dimensions
    width, height = 416, 416
    resized_frame = cv2.resize(frame, (width, height))

    # Perform object detection on the resized frame
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (width, height), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    # Initialize lists to store bounding box coordinates, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the detected objects
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions by ensuring the detected probability is greater than a minimum threshold
            if confidence > 0.5:
                # Scale the bounding box coordinates to match the original frame size
                x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])

                # Calculate the coordinates of the top-left corner
                x_min, y_min = int(x - w / 2), int(y - h / 2)

                # Append the bounding box coordinates, confidences, and class IDs to their respective lists
                boxes.append([x_min, y_min, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any object detections remain after non-maximum suppression
    if len(indices) > 0:
        for i in indices.flatten():
            # Extract bounding box coordinates and class ID
            x_min, y_min, w, h = boxes[i]
            class_id = class_ids[i]

            # Draw the bounding box and label on the frame
            color = (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), color, 2)
            text = f"{classes[class_id]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
