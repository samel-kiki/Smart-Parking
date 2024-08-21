import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Function to print mouse coordinates on mouse movement
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a window for displaying the frame
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture('parking1.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read class names from list.txt
my_file = open("list.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Define parking areas
area1 = [(52, 364), (30, 417), (73, 412), (88, 369)]
area2 = [(105, 353), (86, 428), (137, 427), (146, 358)]
area3 = [(159, 354), (150, 427), (204, 425), (203, 353)]
area4 = [(217, 352), (219, 422), (273, 418), (261, 347)]
area5 = [(274, 345), (286, 417), (338, 415), (321, 345)]
area6 = [(336, 343), (357, 410), (409, 408), (382, 340)]
area7 = [(396, 338), (426, 404), (479, 399), (439, 334)]
area8 = [(458, 333), (494, 397), (543, 390), (495, 330)]
area9 = [(511, 327), (557, 388), (603, 383), (549, 324)]
area10 = [(564, 323), (615, 381), (654, 372), (596, 315)]
area11 = [(616, 316), (666, 369), (703, 363), (642, 312)]
area12 = [(674, 311), (730, 360), (764, 355), (707, 308)]

# Combine all areas into a list for easier processing
areas = [area1, area2, area3, area4, area5, area6, area7, area8, area9, area10, area11, area12]

# Create a window for displaying available spaces
cv2.namedWindow('Available Spaces')

# Start reading the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add a delay to process frames slower
    time.sleep(1)  

    # Resize the frame
    frame = cv2.resize(frame, (1020, 500))  

    # Predict using the YOLO model
    results = model.predict(frame)  
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Initialize a list to count cars in each area
    area_counts = [0] * len(areas)  

    # List to store indices of occupied areas
    occupied_areas = [] 

    # Iterate over the detected objects
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row[:6])
        c = class_list[d]
        if 'car' in c:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

  # Check if the car is in any of the defined parking areas
            for i, area in enumerate(areas):
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    area_counts[i] += 1
                    cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    # Store the occupied area index (1-based index)
                    occupied_areas.append(i + 1)  
                    break

    # Determine available areas
    available_areas = [i + 1 for i in range(len(areas)) if i + 1 not in occupied_areas]  # 1-based index
    print(f"Available parking lots: {available_areas}")

    # Draw polygons and labels for parking areas
    for i, area in enumerate(areas):
        color = (0, 0, 255) if area_counts[i] > 0 else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, str(i + 1), (area[0][0] - 10, area[0][1] + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    #cv2.putText(frame, f"Available: {len(available_areas)}", (23, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("RGB", frame)

    # Display available parking spaces in a separate window
    display_frame = np.zeros((500, 400, 3), np.uint8)
    cv2.putText(display_frame, "Available Spaces:", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for idx, area in enumerate(available_areas):
        cv2.putText(display_frame, f"{area}", (10, 100 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Available Spaces", display_frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()