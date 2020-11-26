import cv2 as cv
import numpy as np
import socket

UDP_IP = '127.0.0.1'
UDP_PORT = 5065

background = None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# Running average between the background and the current frame
def find_avg(img, aweight):
    global background
    if background is None:
        background = img.copy().astype('float')
        return
    cv.accumulateWeighted(img, background, aweight)


def segmentation(image, threshold=25):
    global background

    # Difference between the background and the current frame
    diff = cv.absdiff(image, background.astype('uint8'))

    _, th = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(
        th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        sg = max(contours, key=cv.contourArea)
        return(th, sg)

    else:
        return


if __name__ == "__main__":
    a_weight = 0.5
    n_frames = 0
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    # ROI coordinates
    top, right, bottom, left = 10, 350, 255, 590

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame = cv.flip(frame, 1)
            copy_frame = frame.copy()
            height, width = frame.shape[:2]

            # Region of Interest
            roi = frame[top:bottom, right:left]

            grayFrame = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            grayFrame = cv.GaussianBlur(grayFrame, (7, 7), 0)

            if n_frames < 30:
                find_avg(grayFrame, a_weight)
            else:
                print('avg:30 frames')
                foreground = segmentation(grayFrame)
                # is the foreground segmented?
                if foreground is not None:
                    (th, sg) = foreground
                    cv.drawContours(
                        copy_frame, [sg + (right, top)], -1, (0, 0, 255))

                    # Draw the convex Hull
                    hull = cv.convexHull(sg, returnPoints=False)
                    cv.drawContours(
                        copy_frame, [hull + (right, top)], -1, (255, 0, 0))

                    defects = cv.convexityDefects(sg, hull)
                    if defects is not None:
                        cnt = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(sg[s][0])
                            end = tuple(sg[e][0])
                            far = tuple(sg[f][0])
                            start_x, start_y = start
                            end_x, end_y = end
                            far_x, far_y = far

                            cv.line(copy_frame, (start_x + right, start_y + top),
                                    (end_x + right, end_y + top), (0, 255, 0), 2)
                            cv.circle(
                                copy_frame, (far_x + right, far_y+top), 5, (255, 255, 0), -1)

                    cv.imshow('Threshold', th)

            cv.rectangle(copy_frame, (left, top),
                         (right, bottom), (0, 255, 0), 2)
            n_frames += 1

            cv.imshow('Frame', copy_frame)

            if cv.waitKey(1) == ord('q'):
                break
        else:
            break

cap.release()
cv.destroyAllWindows()
