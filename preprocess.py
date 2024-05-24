import cv2
import numpy as np


def preprocess(path: str):
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    # Performing threshold on the hue channel `hsv[:,:,0]`
    label_thresh = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    box_thresh = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    box_image = cv2.bitwise_and(box_thresh, label_thresh)
    return box_image, image


def draw_squares(box_image, image):
    squares = find_squares(box_image)
    squares = sorted(squares, key=cv2.contourArea, reverse=True)
    squares = squares[-2:-1] + squares[:1]
    cv2.drawContours(image, squares, -1, (255, 0, 0), 1)
    return squares


def show_image(image, name: str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    imS = cv2.resize(image, (800, 1000))  # Resize image
    cv2.imshow(name, imS)  # Show image
    cv2.waitKey(0)


def find_squares(img):
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contourLength = cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, 0.02 * contourLength, True)
                if len(contour) == 4 and cv2.contourArea(contour) > 10000 and cv2.isContourConvex(contour):
                    contour = contour.reshape(-1, 2)
                    max_cos = np.max(
                        [angle_cos(contour[i], contour[(i + 1) % 4], contour[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.19:
                        squares.append(contour)
    return squares


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def perspective_transform(image, bounds):
    p_x, p_y, p_w, p_h = cv2.boundingRect(bounds)
    labelDestination = np.array([[p_w, 0], [0, 0], [0, p_h], [p_w, p_h]], np.float32)
    perspective = cv2.getPerspectiveTransform(bounds.astype(np.float32), labelDestination)
    result = cv2.warpPerspective(image, perspective, (p_w, p_h))
    show_image(result, "perspective transformed image")


if __name__ == '__main__':
    box_img, img = preprocess("static/package_images/IMG_7655.jpeg")
    squares = draw_squares(box_img, img)
    label_bounds, box_bounds = squares[0], squares[1]
    show_image(img, "output")
    x, y, w, h = cv2.boundingRect(label_bounds)
    label = img[y:y + h, x:x + w]
    show_image(label, "label")
    box = img[y:y + h, x:x + w]
    show_image(box, "box")
    perspective_transform(img, box_bounds)
    x, y, w, h = cv2.boundingRect(box_bounds)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
