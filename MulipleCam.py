#!/usr/bin/env python3
import time
import os
import socket
import threading

import mediapipe_utils
from BlazeposeRenderer import BlazeposeRenderer
import argparse
from math import atan2, degrees
from mediapipe_utils import KEYPOINT_DICT
from datetime import datetime, timedelta

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
client_count = 0
client_socket = 0

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)



def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    global client_socket
    global client_count

    client_socket = conn
    client_count = client_count + 1

    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False

            print(f"[{addr}] {msg}")
            conn.send("Msg received".encode(FORMAT))

    conn.close()


def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while client_count < 1:
        conn, addr = server.accept()
        global client_s
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

#print("[STARTING] server is starting...")
#start()


def angle_with_y(v):
    # v: 2d vector (x,y)
    # Returns angle in degree of v with y-axis of image plane
    if v[1] == 0:
        return 90
    angle = atan2(v[0], v[1])
    return degrees(angle)


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
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
else:
    from BlazeposeDepthai import BlazeposeDepthai

tracker = BlazeposeDepthai(input_src=args.input,
            pd_model=args.pd_m,
            lm_model="lite",
            smoothing=not args.no_smoothing,   
            xyz=args.xyz,            
            crop=args.crop,
            internal_fps=15,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=True,
            trace=args.trace)


path = 'results'

isExist = os.path.exists(path)

if not isExist:
    os.makedirs(path)

timestr = time.strftime("%Y%m%d-%H%M%S")
videoFName = path + os.path.sep + timestr + '.avi'
landmarkFName = path + os.path.sep + timestr + '.csv'
landmarkLineFName = path + os.path.sep + 'land_mark_line.csv'

renderer = BlazeposeRenderer(
                tracker,
                show_3d=args.show_3d,
                output=videoFName)

renderer.turnOffLandMarks()

ini_time_for_now = datetime.now()

f = open(landmarkFName, "w")
f.write("Date" + str(ini_time_for_now) + "\n")

f_line = open(landmarkLineFName, "w")


header = 'time'
for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
    header = header + ';' + landMarkPosition + '_x' + ';' + landMarkPosition + '_y'

f.write(header + "\n")

startTime = time.time()

lKeyPressed = False

while True:


    elapsedTime = round(time.time() - startTime, 2)
    lineToWrite = str(round(elapsedTime, 2))



    # Run blazepose on next frame
    frame, body = tracker.next_frame()
    if frame is None: break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    renderer.setElapsedTime(elapsedTime)



    if not(body is None):
        #right_arm_angle = angle_with_y(
        #    body.landmarks[KEYPOINT_DICT['right_elbow'], :2] - body.landmarks[KEYPOINT_DICT['right_shoulder'], :2])
        #left_arm_angle = angle_with_y(
        #    body.landmarks[KEYPOINT_DICT['left_elbow'], :2] - body.landmarks[KEYPOINT_DICT['left_shoulder'], :2])

        for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
            lm = body.landmarks[KEYPOINT_DICT[landMarkPosition], :2]
            lineToWrite = lineToWrite + ';' + str(round(lm[0])) + ';' + str(round(lm[1]))

    else:
        for landMarkPosition in mediapipe_utils.KEYPOINT_DICT:
            lineToWrite = lineToWrite + ';-1;-1'

    f.write(lineToWrite + "\n")
    f_line.truncate()
    f_line.write((lineToWrite))
    f_line.flush()
    os.fsync(f)

        #client_socket.send(lineToWrite.encode())

    if key == 27 or key == ord('q'):
        break

f.close()
renderer.exit()
tracker.exit()
