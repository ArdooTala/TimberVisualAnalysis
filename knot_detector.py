import cv2
import numpy as np
import tomlkit


def extract_contours(img_file, params):
    # with open("params.toml", "r") as f:
    #     params = tomlkit.load(f)
    #     print(params)

    rgb = cv2.imread(f"images/{img_file}")
    rgb = rgb[
          params["crop"]["min_x"]:params["crop"]["max_x"],
          params["crop"]["min_y"]:params["crop"]["max_y"],
          :]

    stacked = rgb[:]

    if params["general"]["verbose"]:
        print(rgb.shape)
        cv2.imshow("Original", rgb)
        cv2.waitKey(params["general"]["step"])

    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if params["general"]["verbose"]:
        cv2.imshow("BW", img)
        cv2.waitKey(params["general"]["step"])

    img = cv2.GaussianBlur(img,
                           (params["blur"]["kernel_size"], params["blur"]["kernel_size"]),
                           0)
    # img = cv2.medianBlur(blw, 11)
    # img = cv2.blur(blw, (11, 11))
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if params["general"]["verbose"]:
        cv2.imshow("Blur", img)
        cv2.waitKey(params["general"]["step"])

    ret, mask = cv2.threshold(img, params["threshold_background"]["threshold"], 255, cv2.THRESH_BINARY)

    img = cv2.bitwise_and(img, img, mask=mask)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if params["general"]["verbose"]:
        cv2.imshow("Thresholding BG", img)
        cv2.waitKey(params["general"]["step"])

    img = cv2.equalizeHist(img)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if params["general"]["verbose"]:
        cv2.imshow("Histogram Equalization", img)
        cv2.waitKey(params["general"]["step"])

    _, img = cv2.threshold(img, params["threshold_knots"]["threshold"], 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(img, img, mask=mask)
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if params["general"]["verbose"]:
        cv2.imshow("Thresholding", img)
        cv2.waitKey(params["general"]["step"])

    kernel = np.ones(
        (params["opening"]["kernel_size"], params["opening"]["kernel_size"]), np.uint8)
    # img = cv2.erode(img, kernel, iterations=15)
    # img = cv2.dilate(img, kernel, iterations=15)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=params["opening"]["iterations"])
    stacked = np.hstack((stacked, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)))
    if params["general"]["verbose"]:
        cv2.imshow("Opening", img)
        cv2.waitKey(params["general"]["step"])

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outlines, _hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours[0]))

    rgb[img < 1] //= 2
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 3)
    stacked = np.hstack((stacked, rgb))
    if params["general"]["verbose"]:
        cv2.imshow("Result", rgb)
        k = cv2.waitKey(100)
        cv2.destroyAllWindows()
    return stacked, [[outline.tolist() for outline in outlines], [contour.tolist() for contour in contours]]
