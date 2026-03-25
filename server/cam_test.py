import cv2

for i in range(5):
    cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    print(i, cam.isOpened())
    cam.release()
