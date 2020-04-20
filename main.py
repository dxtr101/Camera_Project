import cv2 as cv
import numpy as np
import matplotlib as plt

# Video Stream
video = cv.VideoCapture(0)

# Save File Information
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20, (640, 480))

if not video.isOpened():
    print("Cannot open Camera")
    exit()
while video.isOpened():
    # Frame by Frame capture
    ret, frame = video.read()

    if not ret:
        print("Can't get frame. Exiting...")
        break

    # Processing
    # video_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display Image
    cv.imshow('Grey Video', frame)
    if cv.waitKey(1) == ord('q'):
        break

    # Write Frame to File
    out.write(frame)

# Release capture
video.release()
out.release()
cv.destroyAllWindows()