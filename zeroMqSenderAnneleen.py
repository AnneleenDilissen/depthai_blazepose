import cv2
import zmq
import base64
import cv2
import depthai
import time

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://localhost:5555')

# camera = cv2.VideoCapture(0)  # init the camera
#
# if not camera.isOpened():
#     raise IOError("Cannot open webcam")
#
# while True:
#     try:
#         (grabbed, frame) = camera.read()  # grab the current frame
#         frame = cv2.resize(frame, (640, 480))  # resize the frame
#         encoded, buffer = cv2.imencode('.jpg', frame)
#         footage_socket.send(base64.b64encode(buffer), zmq.NOBLOCK)
#
#     except KeyboardInterrupt:
#         camera.release()
#         cv2.destroyAllWindows()
#         print("\n\nBye bye\n")
#         break

# create a DepthAI pipeline
pipeline = depthai.Pipeline()

# create a node that receives the video feed from the camera
cam_node = pipeline.createColorCamera()
cam_node.setPreviewSize(640, 480)
cam_node.setFps(30)

# create a node that displays the video feed
display_node = pipeline.createXLinkOut()
display_node.setStreamName("display")

# link the camera node to the display node
cam_node.preview.link(display_node.input)

# start the pipeline
with depthai.Device(pipeline, usb2Mode=True) as device:
    # create a window to display the video
    # cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("video", 640, 480)

    # initialize some variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.monotonic()

    while True:
        # get the next frame from the video feed
        in_frame = device.getOutputQueue("display").get()

        # convert the frame to a numpy array
        frame = in_frame.getCvFrame()

        # # calculate FPS
        # frame_count += 1
        # elapsed_time = time.monotonic() - start_time
        # if elapsed_time > 1.0:
        #     fps = frame_count / elapsed_time
        #     frame_count = 0
        #     start_time = time.monotonic()

        # display FPS in the video window
        # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # display the frame
        # cv2.imshow("video", frame)

        encoded, buffer = cv2.imencode('.jpg', frame)
        footage_socket.send(base64.b64encode(buffer), zmq.NOBLOCK)

        # wait for key press to exit
        if cv2.waitKey(1) == ord("q"):
            break

    # release resources
    cv2.destroyAllWindows()
