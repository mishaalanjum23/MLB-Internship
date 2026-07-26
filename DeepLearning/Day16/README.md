## Difference between Sobel, Laplacian, and Canny

* **Sobel:** Detects horizontal and vertical edges separately.
* **Laplacian:** Detects edges in all directions.
* **Canny:** Detects edges more accurately by reducing noise and keeping only the important edges.

## Purpose of Morphological Operations

* **Erosion:** Shrinks white objects and removes small white noise.
* **Dilation:** Expands white objects and fills small gaps.
* **Opening:** Removes small white noise.
* **Closing:** Fills small gaps and connects broken edges.
* **Morphological Gradient:** Highlights the boundaries of objects.
* **Top Hat:** Highlights small bright details.
* **Black Hat:** Highlights small dark details.


## Best Combination

The best results were achieved by:

Converting the image to grayscale.
Applying Gaussian Blur with a 3 × 3 kernel.
Using Canny Edge Detection with thresholds 50 and 150.
Applying the Closing morphological operation with a 7 × 7 kernel and 2 iterations before detecting the document boundary.

These changes improved the document boundary detection compared to the initial results.

## Challenges Faced

The biggest challenge was detecting the complete document boundary. In some images, the document edges blended with the background, making them difficult to detect during edge detection. To improve the results, I reduced the Gaussian Blur kernel size, lowered the Canny threshold values, increased the Closing kernel size, and applied two Closing iterations. These changes improved the output, but the boundary detection is still not fully accurate for all images.
