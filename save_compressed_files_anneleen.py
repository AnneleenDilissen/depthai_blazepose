import cv2
import depthai
import time

# create a DepthAI pipeline
pipeline = depthai.Pipeline()

# create a node that receives the video feed from the camera
cam_node = pipeline.createColorCamera()
cam_node.setPreviewSize(640, 480)
cam_node.setFps(30)

# create a node that displays the video feed
display_node = pipeline.createXLinkOut()
display_node.setStreamName("display")

# create a node that compresses the video feed
encoder_node = pipeline.createVideoEncoder()
encoder_node.setDefaultProfilePreset(30, depthai.VideoEncoderProperties.Profile.H264_HIGH)

# create an output node for the encoder
xout_encoder = pipeline.createXLinkOut()
xout_encoder.setStreamName("video_save")

# link the camera node to the display node
cam_node.preview.link(display_node.input)

cam_node.video.link(encoder_node.input)
encoder_node.bitstream.link(xout_encoder.input)

# out = cv2.VideoWriter('output.mjpeg', cv2.VideoWriter_fourcc('M', 'P', 'G', '1'), 30, (cam_node.getPreviewWidth(), cam_node.getPreviewHeight()))

# start the pipeline
with depthai.Device(pipeline, usb2Mode=True) as device:
    # create a window to display the video
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", 640, 480)

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="video_save", maxSize=30, blocking=True)

    with open('video.H264', 'wb') as videoFile:

        while True:

            videoPacket = q.get()
            vd_packet = videoPacket.getData()
            vd_packet.tofile(videoFile)

            print(vd_packet.shape)
            print("1")
            # get the next frame from the video feed
            in_frame = device.getOutputQueue("display").get()

            # convert the frame to a numpy array
            frame = in_frame.getCvFrame()

            # Write the frame into the file 'output.avi'
            # out.write(frame)

            # display the frame
            cv2.imshow("video", frame)

            # wait for key press to exit
            if cv2.waitKey(1) == ord("q"):
                break

        # release resources
        cv2.destroyAllWindows()

