import cv2

image = cv2.imread("img.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x,y), (x + w, y + h), (255, 0, 0), 2)

    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Detect shapes
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    corners = len(approx)
    circularity = 4 * 3.14159 * area / (perimeter ** 2)

    if corners == 3:
        shape = "Triangle"

    elif corners == 4:
        x2,y2,w2,h2 = cv2.boundingRect(approx) 
        aspect_ratio = w2 / h2
        if 0.95 <= aspect_ratio <= 1.05:
            shape = "Square" 
        else:
            shape = "Rectangle"

    else:
           if circularity > 0.85:
                  shape = "Circle"
           else:
              shape = "Polygon"

    cv2.putText(image, shape, (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

cv2.imshow("final image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()