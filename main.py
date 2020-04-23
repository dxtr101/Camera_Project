from __future__ import print_function
from matplotlib import pyplot as plt
from os.path import join
import cv2 as cv
import numpy as np
import argparse
import time
import sys

def bkg_frame(vid):
    '''
    This function obtains a frame from the video feed and saves it as the background image.
    :param vid: video object
    :return: none
    '''
    ret, frame = vid.read()

    if not ret:
        print("Can't get background image.  Exiting...")
        exit()
    else:
        print("Updated background image.")

    cv.imwrite('background.png', frame)
    return time.time()

def main():
    # Video Stream
    video = cv.VideoCapture(0)

    if not video.isOpened():
        print("Cannot open Camera")
        exit()

    # Configuration
    plt.ion()
    bkg_update_period = 6000  # seconds between background updates
    motion_threshold = 390    # Min histogram value that indicates movement
    motion = False            # Initially, no motion detected.
    file_counter = 0          # File sequence
    frame_counter = 0         # Number of frames saved to specific file
    motion_stop_time = 0      # Track when motion ended
    motion_last_frame = False # Need two frames in a row to call motion true
    motion_record_hold = 3    # Seconds to continue recording after motion stops
    frame_limit = 10000       # NUmber of frames allowed per file
    motion_print = 60         # Print if motion only every (x) seconds
    last_print = 0            # last time motion printed
    time.sleep(1)

    # Get Background Frame
    bkg_time = bkg_frame(video)

    # Save File Information
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    if sys.platform == 'win32':
        out = cv.VideoWriter('output_' + str(file_counter) + '.avi', fourcc, 20, (640, 480))
    elif sys.platform == 'linux':
        output_file = 'output_' + str(file_counter) + '.avi'
        output_path = join('/media/lair_drive/Security_Footage', output_file)
        out = cv.VideoWriter(output_path, fourcc, 20, (640, 480))

    # Parser (Detect Differences From Background)
    parser = argparse.ArgumentParser(description='This program detections motion via background subtraction.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='background.png')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()

    if args.algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2(500, 10.0, True)
    else:
        backSub = cv.createBackgroundSubtractorKNN(500, 400.0, True)

    while video.isOpened():
        loop_time = time.time()
        # Frame by Frame capture
        ret, frame = video.read()

        if not ret:
            print("Can't get frame. Exiting...")
            break

        # Processing
        processed_frame = cv.GaussianBlur(frame, (5,5), 0)
        fgMask = backSub.apply(processed_frame)
        hist_mask = cv.calcHist([processed_frame], [0], fgMask, [256], [0, 256])

        # Determine if Motion Detected
        hist_mask_max = np.amax(hist_mask)
        if hist_mask_max > motion_threshold:
            motion_this_frame = True
        else:
            motion_this_frame = False
            motion_stop_time = time.time()

        if motion_last_frame and motion_this_frame:
            motion = True
        else:
            motion = False

        motion_last_frame = motion_this_frame

        # Plot Histogram
        # plt.clf()
        # plt.axis([0, 256, 0, 5000])
        # plt.plot(hist_mask)
        # plt.draw()
        # plt.pause(0.001)

        # Display Image
        # cv.imshow('Video', frame)
        # cv.imshow('Mask', fgMask)
        # if cv.waitKey(1) == ord('q'):
        #     break

        # Update Background
        if (loop_time - bkg_time) > bkg_update_period:
            bkg_time = bkg_frame(video)

        if motion or ((loop_time - motion_stop_time) < motion_record_hold):
            # Add Time Stamp
            cv.rectangle(frame, (10, 2), (240, 20), (255, 255, 255), -1)
            cv.putText(frame, str(time.asctime()), (15, 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            # Write Frame to File
            frame_counter += 1
            if frame_counter >= frame_limit:
                file_counter += 1
                if sys.platform == 'win32':
                    out = cv.VideoWriter('output_' + str(file_counter) + '.avi', fourcc, 20, (640, 480))
                elif sys.platform == 'linux':
                    output_file = 'output_' + str(file_counter) + '.avi'
                    output_path = join('/media/lair_drive/Security_Footage', output_file)
                    out = cv.VideoWriter(output_path, fourcc, 20, (640, 480))
                frame_counter = 0

            if (loop_time - last_print) > motion_print:
                print('Detected Motion.', time.asctime())
            out.write(frame)

    # Release capture
    video.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()