import cv2
import depthai
import time
import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to test server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.0.149:5555")

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
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", 640, 480)

    # initialize some variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.monotonic()

    # start the video feed
    device.startPipeline()

    while True:
        # get the next frame from the video feed
        in_frame = device.getOutputQueue("display").get()

        # convert the frame to a numpy array
        frame = in_frame.getCvFrame()

        # calculate FPS
        frame_count += 1
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.monotonic()

        # display FPS in the video window
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # display the frame
        cv2.imshow("video", frame)

        socket.send(frame)

        #  Get the reply.
        message = socket.recv()
        print("Received reply %s [ %s ]" % message)

        # wait for key press to exit
        if cv2.waitKey(1) == ord("q"):
            break

    # release resources
    cv2.destroyAllWindows()
