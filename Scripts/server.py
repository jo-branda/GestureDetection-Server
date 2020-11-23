import cv2 as cv
import numpy as np
import socket

UDP_IP = '127.0.0.1'
UDP_PORT = 5065

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('Frame', grayFrame)
        if cv.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
