import cv2
 
# images are in the format of height × width × channels  in numpy or opencv
# Load the image
image = cv2.imread("zenodo 200 with labels (200)/1003-receipt.jpg")
 
# Define new width and height
new_width = 1500
new_height = 1500
 
# 1
# resized_image = cv2.resize(image, (new_width, new_height))
 
#2
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3
# image_low = cv2.GaussianBlur(image, (3,3), 0)
blur_med = cv2.GaussianBlur(image, (7,7), 0)
# image_high = cv2.GaussianBlur(image, (15,15), 0)

# canny_low = cv2.Canny(image_med, 20, 50)
canny_med = cv2.Canny(blur_med, 50, 150)
# canny_high = cv2.Canny(image_med, 150, 250)


cv2.imshow("Med Canny", canny_med)

contours, hierarchy = cv2.findContours(canny_med,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print("Len", len(contours))
contours_ranked = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)
# print("ranked countur", contours_ranked[0], "area 1", cv2.contourArea(contours_ranked[0]))
# print("ranked countur", contours_ranked[1],"area 2",  cv2.contourArea(contours_ranked[1]))
# print("ranked countur", contours_ranked[2], "area 3",  cv2.contourArea(contours_ranked[2]))


for i, contour in enumerate(contours_ranked[:10]):
    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)

    print(
        f"Contour {i}: "
        f"Area={area:.0f}, "
        f"Position=({x}, {y}), "
        f"Width={w}, Height={h}"
    )


cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow("Original Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()