import cv2
import imagezmq

# Instantiate and provide the first publisher address
image_hub = imagezmq.ImageHub(open_port='tcp://169.254.4.100:5555', REQ_REP=False)

while True:  # show received images
    rpi_name, image = image_hub.recv_image()
    cv2.imshow(rpi_name, image)  # 1 window for each unique RPi name
    cv2.waitKey(1)