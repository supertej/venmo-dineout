from skimage.filters import threshold_local
import cv2
import numpy as np


def scan(image, show=True):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = resize(image, height=500)
    gray = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edged = cv2.Canny(gray, 20, 100)

    edged_copy = edged.copy()
    edged_copy = cv2.GaussianBlur(edged_copy, (5, 5), 0)
    cv2.imwrite('edged.jpg', edged)

    #if show:
        #cv2.imshow("Edged", edged)
        #cv2.imshow("Edged blurred", edged_copy)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    (cnts, _) = cv2.findContours(edged_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

    screenCnt = []

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        # approx = np.array(cv2.boundingRect(c))
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        debugging = False
        
        #if debugging:
            #cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            #cv2.imshow("Outline", image)
            #cv2.waitKey(0)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt.__len__() != 0:
        #if show:
            #cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
            #cv2.imwrite('outlined.jpg', image)
            #cv2.imshow("Outline", image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    else:
        warped = orig

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    local_threshold = threshold_local(warped, 251, offset=10)
    warped = warped > local_threshold
    warped = warped.astype("uint8") * 255
    cv2.imwrite('deskewed.jpg', warped)

    return warped


    #if show:
        #cv2.imshow("Original", util.resize(orig, height=650))
        #cv2.imshow("Scanned", util.resize(warped, height=650))
        #cv2.waitKey(0)
    #cv2.imwrite('deskewed.jpg', warped)

def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized



def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def order_points(pts):
    # order will be: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum,
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # the top-right point will have the smallest difference,
    # the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)
