import imutils
import cv2 as cv
import numpy as np

#load image
image = cv.imread("tetris_blocks.png")
cv.imshow("Image", image)
cv.waitKey(0)

#convert image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)
cv.waitKey(0)

#find edges using canny algorithm
edged = cv.Canny(gray, 30, 150)
cv.imshow("Edged", edged)
cv.waitKey(0)

#threshold to distinguish objects from background
thresh = cv.threshold(gray, 225, 225, cv.THRESH_BINARY_INV)[1]
cv.imshow("Thresh", thresh)
cv.waitKey(0)

#find contour
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, 
						cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

cv.drawContours(output, cnts, -1, (240, 0, 159), 3)
text = "I found {} objects!".format(len(cnts))
cv.putText(output, text, (10, 25),  cv.FONT_HERSHEY_SIMPLEX, 0.7,
		  (240, 0, 159), 2)
cv.imshow("Contours", output)
cv.waitKey(0)
