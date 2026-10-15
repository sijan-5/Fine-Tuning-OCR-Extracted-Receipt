# import numpy as np
# from skimage import io
# from skimage.color import rgb2gray
# from skimage.transform import rotate
# from deskew import determine_skew


# image = io.imread("/Users/sijanbhattarai/Downloads/fullDataset/zenodo 200 with labels/external.png")
# # removing 4th alpha channel
# image = image[:, :, :3]
# grayscale = rgb2gray(image)
# angle = determine_skew(grayscale)
# rotated = rotate(image=image, angle=angle, resize=True) * 255
# io.imsave("/Users/sijanbhattarai/Downloads/fullDataset/zenodo 200 with labels/external_rotated.png", rotated.astype(np.uint8))
# print(image)


# Source - https://stackoverflow.com/a/59672620
# Posted by nathancy, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-19, License - CC BY-SA 4.0

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

if __name__ == '__main__':
    image = cv2.imread("/Users/sijanbhattarai/Downloads/fullDataset/zenodo 200 with labels/1058-receipt.jpg")
    angle, corrected = correct_skew(image)
    print('Skew angle:', angle)
    cv2.imshow('corrected', corrected)
    cv2.waitKey()
