import cv2
import zmq
import base64
import numpy as np
import time
import os

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:5555')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, "")

with open("zmq_test.mp4", "wb") as f:
    while True:
        frame = footage_socket.recv_string()
        img = base64.b64decode(frame)
        npimg = np.frombuffer(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)

        # f.write(source)

        cv2.imshow("receiver", source)
        cv2.waitKey(1)

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
            cv2.destroyAllWindows()
            f.close()
            footage_socket.close()
            print("\n\nBye bye\n")
            break


