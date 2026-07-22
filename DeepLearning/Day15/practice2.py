import cv2
import numpy as np

img = cv2.imread("img.jpg")

# Brightness
bright = cv2.convertScaleAbs(img, alpha = 1, beta = 50)
dark = cv2.convertScaleAbs(img, alpha = 1, beta = -50)

# Contrast
high_contrast = cv2.convertScaleAbs(img, alpha = 2, beta = 0)
low_contrast = cv2.convertScaleAbs(img, alpha = 0.5, beta = 0)

# Gaussian Blur
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Median Blur
median = cv2.medianBlur(img, 5)

# Bilateral Filter
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Image Sharpening
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharp = cv2.filter2D(img, -1, kernel)

# Display Results
cv2.imshow("Original", img)
cv2.imshow("Bright", bright)
cv2.imshow("Dark", dark)
cv2.imshow("High Contrast", high_contrast)
cv2.imshow("Low Contrast", low_contrast)
cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Median Blur", median)
cv2.imshow("Bilateral Filter", bilateral)
cv2.imshow("Sharpened", sharp)

cv2.waitKey(0)
cv2.destroyAllWindows()