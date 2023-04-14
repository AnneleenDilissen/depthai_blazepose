import cv2
import zmq
import base64
import depthai
import time
from BlazeposeDepthaiEdge import BlazeposeDepthai
from pathlib import Path
import os
from datetime import datetime, timedelta
import mediapipe_utils
from mediapipe_utils import KEYPOINT_DICT
from Anneleen_test_renderer import Anneleen_test_renderer
from math import atan2, degrees
import argparse

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://192.168.0.149:5555')

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

renderer = Anneleen_test_renderer(tracker)

renderer.turnOnLandMarks()

startTime = time.time()

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

    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    renderer.setElapsedTime(elapsedTime)

    encoded, buffer = cv2.imencode('.jpg', frame)
    footage_socket.send(base64.b64encode(buffer), zmq.NOBLOCK)

    if cv2.waitKey(1) == ord('q'):
        break

renderer.exit()
tracker.exit()
