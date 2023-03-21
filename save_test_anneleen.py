import depthai
import time
import cv2

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
cam_rgb.setFps(30)

video_encoder = pipeline.createVideoEncoder()
video_encoder.setDefaultProfilePreset(cam_rgb.getFps(), depthai.VideoEncoderProperties.Profile.H264_BASELINE)
video_encoder.setFrameRate(30)

# create an XLinkOut node for the video encoder output and set its stream name
video_out = pipeline.createXLinkOut()
video_out.setStreamName("video")

# link the video encoder output to the video_out node
cam_rgb.preview.link(video_encoder.input)
video_encoder.bitstream.link(video_out.input)

# create and start the device
device = depthai.Device(pipeline)
video_output_queue = device.getOutputQueue("video", maxSize=2, blocking=False)

# open a file for writing
with open("output.mp4", "wb") as f:
    start_time = time.monotonic()
    frame_count = 0
    print("1")
    while True:
        # get the next packet from the video output queue
        packet = video_output_queue.get()
        print("2")
        if packet is not None:
            # write the packet data to the file
            f.write(packet.getRaw())

        # calculate the FPS and print it
        elapsed_time = time.monotonic() - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")

        # increment the frame count
        frame_count += 1

        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break

# close the device
device.close()
