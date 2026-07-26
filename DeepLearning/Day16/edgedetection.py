import cv2

image = cv2.imread("img.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply guassian blur
blur = cv2.GaussianBlur(gray, (5,5), 0 )

# sobel edge detection
sobely = cv2.Sobel(blur, cv2.CV_64F, 1, 0) # find vertical edges
sobelx = cv2.Sobel(blur, cv2.CV_64F, 0, 1) # find horizontal edges

sobel = cv2.addWeighted(sobely, 0.5, sobelx, 0.5, 0)

# laplacian edge detection
laplacian = cv2.Laplacian(blur, cv2.CV_64F)

# canny edge detection
canny = cv2.Canny(blur, 100, 200)

cv2.imshow("Original Image", image)
cv2.imshow("Sobel Edge Detection", sobel)
cv2.imshow("Laplacian Edge Detection", laplacian)
cv2.imshow("Canny Edge Detection", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()