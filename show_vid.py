import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to update the frame
def update(frame_number):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        ani.event_source.stop()
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    im.set_data(frame)

# Initialize video capture
cap = cv2.VideoCapture('test_videos/test3.mp4')

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")
else:
    # Read the first frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        plt.figure()
        im = plt.imshow(frame, animated=True)
        ani = FuncAnimation(plt.gcf(), update, interval=50)  # Update interval in milliseconds
        plt.show()

# Release the video capture object
cap.release()
