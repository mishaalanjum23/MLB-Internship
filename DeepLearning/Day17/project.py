import cv2
import os

input_folder = "input_images"
output_folder = "output_images"

os.makedirs(output_folder, exist_ok = True)

for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    contour_image = image.copy()

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      area = cv2.contourArea(contour)
      perimeter = cv2.arcLength(contour, True)

      if area < 10:
         continue
      if perimeter == 0:
         continue

      cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
      cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)

       # Detect shapes
      circularity = 4 * 3.14159 * area / (perimeter ** 2)
      approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
      corners = len(approx)
      x,y,w,h = cv2.boundingRect(approx)
      
      if corners == 3:
         shape = "Triangle"
      
      elif corners == 4:
         aspect_ratio = w / h
         if 0.95 <= aspect_ratio <= 1.05:
             shape = "Square" 
         else:
             shape = "Rectangle"

      elif corners == 5:
         shape = "Pentagon"

      elif corners == 6:
         shape = "Hexagon"

      elif corners == 7:
         shape = "Heptagon"

      else:
        if circularity > 0.85:
               shape = "Circle"
        else:
           shape = "Polygon"

      cv2.putText(image, shape, (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
      cv2.putText(image, f"Area: {int(area)}",(x,y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
      cv2.putText(image, f"Peri: {int(perimeter)}",(x,y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imwrite(os.path.join(output_folder, f"contours_{filename}"), contour_image)
    cv2.imwrite(os.path.join(output_folder, f"final_{filename}"), image)

print("Done") 
