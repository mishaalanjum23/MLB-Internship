import cv2
import numpy as np

image = cv2.imread("img.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# create kernel
kernel = np.ones((5,5), np.uint8)

# Erosion
erosion = cv2.erode(gray, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(gray, kernel, iterations=1)

# Opening
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

# closing
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# Gradient
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

# tophat
top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

# blackhat
black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)
cv2.imshow("Morphological Gradient", gradient)
cv2.imshow("Tophat", top_hat)
cv2.imshow("Blackhat", black_hat)

cv2.waitKey(0)
cv2.destroyAllWindows()