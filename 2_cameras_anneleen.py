import depthai as dai
import threading
import contextlib
import cv2
import time

# This can be customized to pass multiple parameters
def getPipeline(stereo):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # For the demo, just set a larger RGB preview size for OAK-D
    if stereo:
        cam_rgb.setPreviewSize(600, 300)
    else:
        cam_rgb.setPreviewSize(600, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)

    # Create output
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    return pipeline

def worker(dev_info, stack, dic):
    openvino_version = dai.OpenVINO.Version.VERSION_2021_4
    device: dai.Device = stack.enter_context(dai.Device(openvino_version, dev_info, True))

    # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
    print("=== Connected to " + dev_info.getMxId())
    mxid = device.getMxId()
    cameras = device.getConnectedCameras()
    usb_speed = device.getUsbSpeed()
    print("   >>> MXID:", mxid)
    print("   >>> Cameras:", *[c.name for c in cameras])
    print("   >>> USB speed:", usb_speed.name)

    device.startPipeline(getPipeline(len(cameras)==3))
    dic["rgb-" + mxid] = device.getOutputQueue(name="rgb")

device_infos = dai.Device.getAllAvailableDevices()
print(f'Found {len(device_infos)} devices')

with contextlib.ExitStack() as stack:
    queues = {}
    threads = []
    for dev in device_infos:
        time.sleep(1) # Currently required due to XLink race issues
        thread = threading.Thread(target=worker, args=(dev, stack, queues))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join() # Wait for all threads to finish

    frame_count = 0

    while True:
        # calculate frame count
        frame_count += 1
        for name, queue in queues.items():
            if queue.has():
                print(queue)
                frame = queue.get().getCvFrame()
                # display frame count in the video window
                cv2.putText(frame, f"frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow(name, frame)


        if cv2.waitKey(1) == ord('q'):
            break

print('Devices closed')