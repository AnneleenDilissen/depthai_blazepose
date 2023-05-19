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

def angle_with_y(v):
    # v: 2d vector (x,y)
    # Returns angle in degree of v with y-axis of image plane
    if v[1] == 0:
        return 90
    angle = atan2(v[0], v[1])
    return degrees(angle)

SCRIPT_DIR = Path(__file__).resolve().parent

# tracker = BlazeposeDepthai(input_src="rgb",
#             pp_model=str(SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob"),
#             lm_model="lite",
#             smoothing=False,
#             xyz=False,
#             crop=False,
#             internal_fps=25,
#             internal_frame_height=640,
#             force_detection=True,
#             stats=True,
#             trace=False)

import argparse
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

tracker = BlazeposeDepthai(input_src=args.input,
                           pd_model=args.pd_m,
                           lm_model="lite",
                           smoothing=not args.no_smoothing,
                           xyz=args.xyz,
                           crop=args.crop,
                           internal_fps=25,
                           internal_frame_height=args.internal_frame_height,
                           force_detection=args.force_detection,
                           stats=True,
                           trace=args.trace,
                           manualfocus=False,
                           manualfocusvalue=0,
                           manualexposure=False,
                           manualexposurevalue=0
                           )



# path = 'results'
#
# isExist = os.path.exists(path)
#
# if not isExist:
#     os.makedirs(path)
#
# timestr = time.strftime("%Y%m%d-%H%M%S")
# videoFName = path + os.path.sep + timestr + '.avi'
# landmarkFName = path + os.path.sep + timestr + '.csv'
# landmarkLineFName = path + os.path.sep + 'land_mark_line.csv'
# ini_time_for_now = datetime.now()

renderer = Anneleen_test_renderer(
                tracker,
                #show_3d=args.show_3d,
                # output=videoFName
                )

renderer.turnOnLandMarks()

# f = open(landmarkFName, "w")
# f.write("Date" + str(ini_time_for_now) + "\n")
#
# f_line = open(landmarkLineFName, "w")
#
#
# header = 'time'
# for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
#     header = header + ';' + landMarkPosition + '_x' + ';' + landMarkPosition + '_y'
#
# f.write(header + "\n")

startTime = time.time()

# lKeyPressed = False

frame_count = 0

while True:

    # calculate frame count
    frame_count += 1

    elapsedTime = round(time.time() - startTime, 2)

    # Run blazepose on next frame
    frame, body = tracker.next_frame()
    cv2.putText(frame, f"frame: {frame_count}", (680, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
    if frame is None:
        break

    #cv2.imshow('Frame', frame)# Draw 2d skeleton

    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    renderer.setElapsedTime(elapsedTime)

    # if not(body is None):
    #     #     right_arm_angle = angle_with_y(
    #     #         body.landmarks[KEYPOINT_DICT['right_elbow'], :2] - body.landmarks[KEYPOINT_DICT['right_shoulder'], :2])
    #     #     left_arm_angle = angle_with_y(
    #     #         body.landmarks[KEYPOINT_DICT['left_elbow'], :2] - body.landmarks[KEYPOINT_DICT['left_shoulder'], :2])
    #     #
    #     #     for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
    #     #         lm = body.landmarks[KEYPOINT_DICT[landMarkPosition], :2]
    #     #         lineToWrite = lineToWrite + ';' + str(round(lm[0])) + ';' + str(round(lm[1]))
    #     #
    #     # else:
    #     #     for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
    #     #         lineToWrite = lineToWrite + ';-1;-1'

    # f.write(lineToWrite + "\n")
    # f_line.truncate()
    # f_line.write(lineToWrite)
    # f_line.flush()
    # os.fsync(f)



    if cv2.waitKey(1) == ord('q'):
        break

# f.close()
renderer.exit()
tracker.exit()