import cv2
import numpy as np


def extract_contours(img_file, params, verbose=False):
    rgb = cv2.imread(f"images/{img_file}")
    rgb = rgb[
          params["crop"]["min_x"]:params["crop"]["max_x"],
          params["crop"]["min_y"]:params["crop"]["max_y"],
          :]

    stacked = rgb[:]
    if verbose:
        print(rgb.shape)
        cv2.imshow("Original", rgb)
        cv2.waitKey(0)

    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("BW", img)
        cv2.waitKey(0)

    img = cv2.GaussianBlur(img,
                           (params["blur"]["kernel_size"], params["blur"]["kernel_size"]),
                           0)
    # img = cv2.medianBlur(blw, 11)
    # img = cv2.blur(blw, (11, 11))
    
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Blur", img)
        cv2.waitKey(0)

    ret, mask = cv2.threshold(img, params["threshold_background"]["threshold"], 255, cv2.THRESH_BINARY)

    img = cv2.bitwise_and(img, img, mask=mask)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Thresholding BG", img)
        cv2.waitKey(0)

    img = cv2.equalizeHist(img)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Histogram Equalization", img)
        cv2.waitKey(0)

    _, img = cv2.threshold(img, params["threshold_knots"]["threshold"], 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(img, img, mask=mask)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Thresholding", img)
        cv2.waitKey(0)

    kernel = np.ones(
        (params["opening"]["kernel_size"], params["opening"]["kernel_size"]), np.uint8)
    # img = cv2.erode(img, kernel, iterations=15)
    # img = cv2.dilate(img, kernel, iterations=15)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=params["opening"]["iterations"])
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Opening", img)
        cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outlines, _hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours[0]))

    rgb[img < 1] //= 2
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 3)
    stacked = np.hstack((stacked, rgb))
    if verbose:
        cv2.imshow("Result", rgb)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
    return stacked, [[outline.tolist() for outline in outlines], [contour.tolist() for contour in contours]]


def grain_direction(img_file, params, verbose=False):
    rgb = cv2.imread(f"images/{img_file}")
    rgb = rgb[
          params["crop"]["min_x"]:params["crop"]["max_x"],
          params["crop"]["min_y"]:params["crop"]["max_y"],
          :]

    stacked = rgb[:]
    if verbose:
        print(rgb.shape)
        cv2.imshow("Original", rgb)
        cv2.waitKey(0)

    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("BW", img)
        cv2.waitKey(0)

    img = cv2.GaussianBlur(img,
                           (params["blur"]["kernel_size"], params["blur"]["kernel_size"]),
                           0)
    # img = cv2.medianBlur(blw, 11)
    # img = cv2.blur(blw, (11, 11))

    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Blur", img)
        cv2.waitKey(0)

    ret, mask = cv2.threshold(img, params["threshold_background"]["threshold"], 255, cv2.THRESH_BINARY)

    img = cv2.bitwise_and(img, img, mask=mask)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Thresholding BG", img)
        cv2.waitKey(0)

    img = cv2.equalizeHist(img)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if verbose:
        cv2.imshow("Histogram Equalization", img)
        cv2.waitKey(0)

    filtered_blurred_x = cv2.Sobel(img, cv2.CV_32F, 1, 0,
                                   ksize=params["gradients"]["kernel_size"],
                                   borderType=cv2.BORDER_DEFAULT)
    filtered_blurred_y = cv2.Sobel(img, cv2.CV_32F, 0, 1,
                                   ksize=params["gradients"]["kernel_size"],
                                   borderType=cv2.BORDER_DEFAULT)

    ori = np.arctan2(filtered_blurred_x, filtered_blurred_y) / np.pi
    ori[ori < 0] += 1
    mag = cv2.magnitude(filtered_blurred_x, filtered_blurred_y)

    mag_downsampled = cv2.resize(mag, None,
                                 fx=1 / params["downsample"]["factor"],
                                 fy=1 / params["downsample"]["factor"])
    mag_normalized = cv2.normalize(mag_downsampled, None, alpha=1, beta=0, norm_type=cv2.NORM_INF)
    ori_downsampled = cv2.resize(ori, None,
                                 fx=1 / params["downsample"]["factor"],
                                 fy=1 / params["downsample"]["factor"])

    hsv = np.zeros_like(rgb)
    hsv[..., 0] = ori * 180
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    stacked = np.hstack((stacked, bgr))
    if verbose:
        cv2.imshow("Gradiant Directions", bgr)
        cv2.waitKey(0)

    bgr = cv2.medianBlur(bgr, params["gradients_blur"]["kernel_size"])

    stacked = np.hstack((stacked, bgr))
    if verbose:
        cv2.imshow("Smoothed Gradiant Directions", bgr)
        cv2.waitKey(0)

    return stacked, [ori_downsampled.shape, ori_downsampled.tolist(), mag_normalized.tolist()]


if __name__ == "__main__":
    params = {
        "crop": {
            "min_x": 100,
            "max_x": 1200,
            "min_y": 400,
            "max_y": 700,
        },
        "blur": {
            "kernel_size": 15
        },
        "threshold_background": {
            "threshold": 127
        },
        "gradients": {
            "kernel_size": 31
        },
        "gradients_blur": {
            "kernel_size": 31
        },
        "downsample": {
            "factor": 2
        }
    }
    grain_direction("NQPQ1107.JPG", params, True)
