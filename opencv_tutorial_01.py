# USAGE
# python opencv_tutorial_01.py

# import the necessary packages
import imutils
import cv2

# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
image = cv2.imread("jp.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

# accessing the pixel values
(B, G, R) = image[100, 50]

print("R : {}, G : {}, B : {}".format(R, G, B))

# Getting Region of interest
roi = image[60:160, 320:420]

cv2.imshow("ROI", roi)
cv2.waitKey(0)
