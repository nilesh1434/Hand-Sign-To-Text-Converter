import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#initialize webcam
cap = cv2.VideoCapture(0)

# create a HandDetector object to detect a maximum of one hand
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/A"
# counter to track the number of saved images
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        # extract the bounding box coordinates (x, y) and dimensions (width, height)
        x, y, w, h = hand["bbox"]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        # if the hand is taller than it is wide
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            # calculate the horizontal gap to center the image
            hGap = math.ceil((imgSize - hCal) / 2)
            # place the resized image in the center of the white background
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # display the cropped hand image
        cv2.imshow("ImageCrop", imgCrop)
        # display the processed white-background hand image
        cv2.imshow("ImageWhite", imgWhite)

    # display the original image with hand detection
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # if the 's' key is pressed, save the processed image
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
