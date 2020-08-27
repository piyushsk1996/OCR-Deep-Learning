import imutils
import cv2
import os

path = './Cropped_Images/'

for root, dirs, files in os.walk("./Images_from_pdf/"):
    for file in files:
        image = cv2.imread('./Images_from_pdf/' + str(file))

        resized = imutils.resize(image, width=1600, height=900)

        ROI = resized[0:400, 0:1600]
        filename = str(file).split('.')[0]
        path = './Cropped_Images/'
        cv2.imwrite(str(path) + filename + '.jpg', ROI)
