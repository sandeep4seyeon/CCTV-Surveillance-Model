import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from models import Darknet
from utils.datasets import LoadImagesAndLabels
from utils.utils import *

# Function to extract frames from a video file
def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize video capture object
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the current frame as an image file
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved: {frame_path}")

        frame_count += 1

    # Release the video capture object
    cap.release()

# Define the path to your video and output folder
video_path = "path_to_your_video_file.mp4"
output_folder = "output_frames"

# Extract frames without showing or closing GUI windows
extract_frames(video_path, output_folder)

video_path = "path to video.mp4"
output_folder = "/path/to/output/frames"
extract_frames(video_path, output_folder)

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def detect_objects_in_frame(frame):
    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Extract object detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return class_ids, confidences, boxes

# Example features extracted from video (e.g., number of people detected, time, duration)
features = np.array([[3, 12, 5], [5, 18, 7], [1, 22, 2], [8, 3, 15], [2, 10, 3]])

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(features_scaled)

# Assign clusters
clusters = kmeans.predict(features_scaled)
print(f"Cluster assignments: {clusters}")

def generate_alert(cluster):
    if cluster == 0:
        print("Alert: Theft detected!")
    elif cluster == 1:
        print("Alert: Arson detected!")
    elif cluster == 2:
        print("Alert: Vandalism detected!")


# Initialize video feed
cap = cv2.VideoCapture("path_to_your_video_file.mp4")

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize variables for time tracking
event_start_time = None
event_duration = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current frame index
    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Calculate the current time in the video
    current_time = frame_index / fps

    # Perform your detection here (assuming you already have detection logic)
    class_ids, confidences, boxes = detect_objects_in_frame(frame)

    # Example: Assume event starts when a person is detected
    if len(class_ids) > 0:
        # If event hasn't started yet, mark the start time
        if event_start_time is None:
            event_start_time = current_time

        # Calculate the event duration
        event_duration = current_time - event_start_time

    # Display the current time and event duration
    print(
        f"Current time: {current_time:.2f}s, Event duration: {event_duration:.2f}s" if event_duration else "No event detected")

    # Show the video (optional)
    cv2.imshow('CCTV Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Example cluster results
for cluster in clusters:
    generate_alert(cluster)

def process_live_video(video_feed):
    cap = cv2.VideoCapture(video_feed)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        class_ids, confidences, boxes = detect_objects_in_frame(frame)

        # Extract features (e.g., number of people)
        num_people = class_ids.count(classes.index('person'))

        # Generate real-time alerts based on clustering
        feature_vector = np.array([num_people, current_time, event_duration])
        feature_vector_scaled = scaler.transform([feature_vector])
        cluster = kmeans.predict(feature_vector_scaled)
        generate_alert(cluster[0])

    cap.release()
    cv2.destroyAllWindows()
