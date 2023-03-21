import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()

mono = pipeline.createMonoCamera()
mono.setBoardSocket(dai.CameraBoardSocket.LEFT)

xout = pipeline.createXLinkOut()
xout.setStreamName("left")
mono.out.link(xout.input)
with dai.Device(pipeline) as device:
    queue = device.getOutputQueue(name="left")
    frame = queue.get()
    imOut = frame.getCvFrame()

def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
      # Convert frame to OpenCV format and return
    return frame.getCvFrame()

