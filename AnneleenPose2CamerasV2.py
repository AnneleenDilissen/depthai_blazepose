from BlazeposeDepthaiEdge import BlazeposeDepthai
from pathlib import Path
import time
import os
from datetime import datetime, timedelta
import mediapipe_utils
from mediapipe_utils import KEYPOINT_DICT
import cv2
from Anneleen_test_renderer import Anneleen_test_renderer
from math import atan2, degrees
import argparse
import depthai as dai
import configparser

def angle_with_y(v):
    # v: 2d vector (x,y)
    # Returns angle in degree of v with y-axis of image plane
    if v[1] == 0:
        return 90
    angle = atan2(v[0], v[1])
    return degrees(angle)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--edge', action="store_true",
                        help="Use Edge mode (postprocessing runs on the device)")
    parser_tracker = parser.add_argument_group("Tracker arguments")
    parser_tracker.add_argument('-i', '--input', type=str, default="rgb",
                                help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
    parser_tracker.add_argument("--pd_m", type=str,
                                help="Path to an .blob file for pose detection model")
    parser_tracker.add_argument("--lm_m", type=str,
                                help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
    parser_tracker.add_argument('-xyz', '--xyz', action="store_true",
                                help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
    parser_tracker.add_argument('-c', '--crop', action="store_true",
                                help="Center crop frames to a square shape before feeding pose detection model")
    parser_tracker.add_argument('--no_smoothing', action="store_true",
                                help="Disable smoothing filter")
    parser_tracker.add_argument('-f', '--internal_fps', type=int,
                                help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")
    parser_tracker.add_argument('--internal_frame_height', type=int, default=640,
                                help="Internal color camera frame height in pixels (default=%(default)i)")
    parser_tracker.add_argument('-s', '--stats', action="store_true",
                                help="Print some statistics at exit")
    parser_tracker.add_argument('-t', '--trace', action="store_true",
                                help="Print some debug messages")
    parser_tracker.add_argument('--force_detection', action="store_true",
                                help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

    parser_renderer = parser.add_argument_group("Renderer arguments")
    parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                                 help="Display skeleton in 3d in a separate window. See README for description.")
    parser_renderer.add_argument("-o", "--output",
                                 help="Path to output video file")

    args = parser.parse_args()

    return args


def main():
    config = configparser.ConfigParser()
    config.read('pose_settings/settings.ini')

    device_infos = dai.Device.getAllAvailableDevices()

    print(len(device_infos))

    save_video = config['DEFAULT']['save_video']
    save_landmarks = config['DEFAULT']['save_landmarks']
    landmark_model = config['DEFAULT']['landmark_model']
    manualfocus = config['DEFAULT']['manualfocus']
    manualfocusvalue = config['DEFAULT']['manualfocusvalue']
    manualexposure = config['DEFAULT']['manualexposure']
    manualexposurevalue = config['DEFAULT']['manualexposurevalue']

    print(save_video)
    print(save_landmarks)


    args = argument_parser()

    for i in range(len(device_infos)):
        temp_tracker = f"tracker_{i}"
        temp_renderer = f"renderer_{i}"
        globals()[temp_tracker] = BlazeposeDepthai(input_src=args.input,
                                                  pd_model=args.pd_m,
                                                  lm_model=landmark_model,
                                                  smoothing=not args.no_smoothing,
                                                  xyz=args.xyz,
                                                  crop=args.crop,
                                                  internal_fps=25,
                                                  internal_frame_height=args.internal_frame_height,
                                                  force_detection=args.force_detection,
                                                  stats=True,
                                                  trace=args.trace,
                                                  manualfocus=manualfocus,
                                                  manualfocusvalue=manualfocusvalue,
                                                  manualexposure=manualexposure,
                                                  manualexposurevalue=manualexposurevalue
                                                  )

        path = 'results'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        timestr = time.strftime("%Y%m%d-%H%M%S")

        if save_landmarks == True:
            print("here")
            temp_f = f"f_{i}"
            landmarkFName = path + os.path.sep + timestr + f"_camera_{i}" + '.csv'
            landmarkLineFName = path + os.path.sep + timestr + f"_camera_{i}" + '_land_mark_line.csv'
            ini_time_for_now = datetime.now()

            globals()[temp_f] = open(landmarkFName, "w")
            globals()[temp_f].write("Date" + str(ini_time_for_now) + "\n")

            temp_f_line = f"f_line_{i}"
            globals()[temp_f_line] = open(landmarkLineFName, "w")

            header = 'time'
            for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
                header = header + ';' + landMarkPosition + '_x' + ';' + landMarkPosition + '_y'

            globals()[temp_f].write(header + "\n")

        if save_video == True:
            videoFName = path + os.path.sep + timestr + f"_camera_{i}" + '.avi'
        else:
            videoFName = None

        globals()[temp_renderer] = Anneleen_test_renderer(globals()[temp_tracker], videoFName, i)
        globals()[temp_renderer].turnOnLandMarks()

        print("=== Connected to " + device_infos[i].getMxId())
        mxid = globals()[temp_tracker].device.getMxId()
        cameras = globals()[temp_tracker].device.getConnectedCameras()
        usb_speed = globals()[temp_tracker].device.getUsbSpeed()
        print("   >>> MXID:", mxid)
        print("   >>> Cameras:", *[c.name for c in cameras])
        print("   >>> USB speed:", usb_speed.name)

    startTime = time.time()

    frame_count = 0

    while True:

        frame_count += 1

        for i in range(len(device_infos)):

            elapsedTime = round(time.time() - startTime, 2)
            lineToWrite = str(round(elapsedTime, 2))

            temp_f = f"f_{i}"
            temp_f_line = f"f_line_{i}"
            temp_frame = f"frame_cam_{i}"
            temp_body = f"body_cam_{i}"

            # Run blazepose on next frame
            globals()[temp_frame], globals()[temp_body] = globals()[f"tracker_{i}"].next_frame()
            cv2.putText(globals()[temp_frame], f"frame: {frame_count}", (680, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
            if globals()[temp_frame] is None:
                break

            globals()[temp_frame] = globals()[f"renderer_{i}"].draw(globals()[temp_frame], globals()[temp_body])
            key = globals()[f"renderer_{i}"].waitKey(delay=1)
            globals()[f"renderer_{i}"].setElapsedTime(elapsedTime)

            if save_landmarks == True:
                if not (globals()[temp_body] is None):
                    right_arm_angle = angle_with_y(
                        globals()[temp_body].landmarks[KEYPOINT_DICT['right_elbow'], :2] - globals()[temp_body].landmarks[KEYPOINT_DICT['right_shoulder'], :2])
                    left_arm_angle = angle_with_y(
                        globals()[temp_body].landmarks[KEYPOINT_DICT['left_elbow'], :2] - globals()[temp_body].landmarks[KEYPOINT_DICT['left_shoulder'], :2])

                    for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
                        lm = globals()[temp_body].landmarks[KEYPOINT_DICT[landMarkPosition], :2]
                        lineToWrite = lineToWrite + ';' + str(round(lm[0])) + ';' + str(round(lm[1]))

                else:
                    for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
                        lineToWrite = lineToWrite + ';-1;-1'

                globals()[temp_f].write(lineToWrite + "\n")
                globals()[temp_f].truncate()
                globals()[temp_f_line].write(lineToWrite)
                globals()[temp_f_line].flush()
                os.fsync(globals()[temp_f])

        if cv2.waitKey(1) == ord('q'):
            break

    for i in range(len(device_infos)):
        globals()[f"renderer_{i}"].exit()
        globals()[f"tracker_{i}"].exit()

        if save_landmarks == True:
            globals()[temp_f].close()

if __name__ == "__main__":
    main()
