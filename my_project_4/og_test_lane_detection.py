import cv2
import numpy as np

def find_edges(image, low_threshold = 70, high_threshold = 150, show_result = False):
    blur = cv2.GaussianBlur(image, (4, 4), 0)
    # Since edge detection is susceptible to noise in the image!
    ' https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html '

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ' https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html '

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    ' https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html '

    if show_result:
        cv2.namedWindow("Canny Edge Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Canny Edge Detection", 1280, 720)
        cv2.imshow("Canny Edge Detection", edges)

    return edges;

def create_mask(edges):
    # ROI mask
    mask = np.zeros_like(edges)
    (height, width) = edges.shape
    polygon = np.array([[
        (0, height), 
        (width, height), 
        (int(.4 * width), int(.65 * height)), 
        (int(.35 * width), int(.65 * height))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    # Fills the area bounded by one or more polygons!

    # cv2.namedWindow("Polygon Mask", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Polygon Mask", 1280, 720)
    # cv2.imshow("Polygon Mask", mask)

    masked_edges = cv2.bitwise_and(edges, mask)

# Ανίχνευση λορίδων
lines = cv2.HoughLinesP(
    masked_edges,
    1,
    (0.5 * np.pi) / 180,
    50,
    minLineLength = 50,
    maxLineGap = 150
)
' https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html '
' https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb '

line_img = np.zeros_like(img)
if lines is not None:
    for line in lines:
        (x1, y1, x2, y2) = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

combo = cv2.addWeighted(original, 0.8, line_img, 1, 0)
' https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html '

cv2.namedWindow("Detected Road Lines", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Road Lines", 1280, 720)
cv2.imshow("Detected Road Lines", combo)

cv2.waitKey(0)
cv2.destroyAllWindows()
def main():
    img = cv2.imread('um_000084.png')
    original = img.copy()

    return;