import cv2
import numpy as np

def nothing(x):
    pass

# Load image
img_path = r'data\04.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
res = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
image = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype("float32")

# Create a window
cv2.namedWindow('Trackbar')
cv2.namedWindow('Trackbar2')
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'Trackbar', 0, 179, nothing)
cv2.createTrackbar('SMin', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('VMin', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('HMax', 'Trackbar', 0, 179, nothing)
cv2.createTrackbar('SMax', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('VMax', 'Trackbar', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'Trackbar', 179)
cv2.setTrackbarPos('SMax', 'Trackbar', 255)
cv2.setTrackbarPos('VMax', 'Trackbar', 255)

cv2.createTrackbar('HMin2', 'Trackbar2', 0, 179, nothing)
cv2.createTrackbar('SMin2', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('VMin2', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('HMax2', 'Trackbar2', 0, 179, nothing)
cv2.createTrackbar('SMax2', 'Trackbar2', 0, 255, nothing)
cv2.createTrackbar('VMax2', 'Trackbar2', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax2', 'Trackbar2', 179)
cv2.setTrackbarPos('SMax2', 'Trackbar2', 255)
cv2.setTrackbarPos('VMax2', 'Trackbar2', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'Trackbar')
    sMin = cv2.getTrackbarPos('SMin', 'Trackbar')
    vMin = cv2.getTrackbarPos('VMin', 'Trackbar')
    hMax = cv2.getTrackbarPos('HMax', 'Trackbar')
    sMax = cv2.getTrackbarPos('SMax', 'Trackbar')
    vMax = cv2.getTrackbarPos('VMax', 'Trackbar')

    hMin2 = cv2.getTrackbarPos('HMin2', 'Trackbar2')
    sMin2 = cv2.getTrackbarPos('SMin2', 'Trackbar2')
    vMin2 = cv2.getTrackbarPos('VMin2', 'Trackbar2')
    hMax2 = cv2.getTrackbarPos('HMax2', 'Trackbar2')
    sMax2 = cv2.getTrackbarPos('SMax2', 'Trackbar2')
    vMax2 = cv2.getTrackbarPos('VMax2', 'Trackbar2')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    lower2 = np.array([hMin2, sMin2, vMin2])
    upper2 = np.array([hMax2, sMax2, vMax2])

    # Convert to HSV format and color threshold
    mask1 = cv2.inRange(image, lower, upper)
    mask2 = cv2.inRange(image, lower2, upper2)
    mask = mask1 + mask2
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(res, res, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

cv2.destroyAllWindows()