import cv2
import numpy as np

# Set up video capture
cap = cv2.VideoCapture(0)

# Define range of blue color in RGB
lower_blue = np.array([0, 60, 80])
upper_blue = np.array([60, 100, 255])

while True:
    # Read in frame from video feed
    ret, frame = cap.read()

    # Convert frame to RGB color space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a mask with only blue pixels
    mask = cv2.inRange(rgb, lower_blue, upper_blue)

    # Apply a blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours of blue areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw boxes around blue areas
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # only draw boxes around large blue areas
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the video feed with blue boxes drawn around detected areas
    cv2.imshow('Blue Boxes', frame)

    # Quit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()