import cv2
import numpy as np

img = cv2.imread("img.jpg")

rows, cols = img.shape[:2]
 
# Translation
M_translate = np.float32([
    [1, 0, 100],  
    [0, 1, 50]     
])
translated = cv2.warpAffine(img, M_translate, (cols, rows))

# 2. Rotation
center = (cols // 2, rows // 2)
rotate45 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 45, 1), (cols, rows))
rotate90 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 90, 1), (cols, rows))

# Scaling
scaled_up = cv2.resize(img, None, fx = 2, fy = 2)
scaled_down = cv2.resize(img, None, fx = 0.5, fy = 0.5)

# Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M_affine = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M_affine, (cols, rows))

# Perspective Transformation
pts1 = np.float32([[50, 50], [250, 50], [50, 250], [250, 250]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M_perspective, (300, 300))

cv2.imshow("Original", img)
cv2.imshow("Translated", translated)
cv2.imshow("Rotate 45", rotate45)
cv2.imshow("Rotate 90", rotate90)
cv2.imshow("Scaled Up", scaled_up)
cv2.imshow("Scaled Down", scaled_down)
cv2.imshow("Affine", affine)
cv2.imshow("Perspective", perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()