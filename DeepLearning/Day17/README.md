# Shape Detection System

## What are Contours?
Contours are the boundaries of objects in an image. They help identify the shape and outline of different objects.

## How Contour Detection Works
The program converts the image to grayscale, applies Gaussian Blur and Canny Edge Detection, then finds contours using OpenCV. The contours are used to detect different shapes based on their corners and circularity.

## Shapes Detected
- Triangle
- Square
- Rectangle
- Circle
- Pentagon
- Hexagon
- Heptagon
- Polygon

## Challenges Faced
One challenge was detecting circles correctly because they were sometimes detected as polygons. I also faced issues with triangle detection due to noisy edges. Using Gaussian Blur and adjusting the contour approximation value improved the detection results. 
