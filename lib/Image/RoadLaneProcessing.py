# Adding road lane lines

import numpy as np
import cv2

LEFT_LANE_START = 15
RIGHT_LANE_START = 210
LANE_HEIGHT = 110
LANE_CENTER = 100


# Generating mask for where the road line boundaries are
def region_of_interest(image):
    height = image.shape[0]
    # Currently, cutting out a "triangle" where lines should be detected
    polygons = np.array([
        [(0, height), (224, height), (110, 100)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    # If the triangle contains too much noise, try using only lines around the area where lane lines should be present
    # p1 = (LEFT_LANE_START, height)
    # p2 = (RIGHT_LANE_START, height)
    # p3 = (LANE_HEIGHT, LANE_CENTER)
    # cv2.line(mask, p2, p3, 255, 30)
    # cv2.line(mask, p1, p3, 255, 30)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def hough_lines(image):
    # TODO: Tune parameters
    return cv2.HoughLinesP(image, 2, np.pi / 180, 46, np.array([]), minLineLength=35,
                           maxLineGap=10)


# Reducing multiple lines into one averaged line
def average_slope_intercept(lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if 0 > slope:
            left_fit.append((slope, intercept))
        elif 0 < slope:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    return left_fit_average, right_fit_average


# Creating coordinates from slopes and intercepts
def make_coordinates(image, line_parameters, height=3 / 5):
    if type(line_parameters) is not np.ndarray:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * height)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# Rendering lines into empty image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is None:
            continue
        x1, y1, x2, y2 = line.reshape(4)

        try:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        except OverflowError:
            # TODO Why is this happening? (And is it still happening?)
            return image
    return line_image


# If 'get_directions' is true, returns whether to steer left or right instead of image
# The action is calculated by comparing center of detected lanes and center of image
def calculate_directions(image, left_line, right_line):
    # Calculating center of found lines
    x1, y1, x2, y2 = left_line.reshape(4)
    x3, y3, x4, y4 = right_line.reshape(4)
    center_x = round((x1 + x2 + x3 + x4) / 4)
    x1, x2, x3 = LEFT_LANE_START, RIGHT_LANE_START, LANE_HEIGHT
    x = round((x1 + x2 + x3) / 3)

    if center_x > x:
        print("Turn right")
        return 'd'
    elif center_x < x:
        print("Turn left")
        return 'a'
    else:
        print("Straight")
        return 'w'

    # Draw center of lane and center of image
    # image = cv2.circle(image, (center_x,center_y), radius=0, color=200, thickness=5)
    # return cv2.circle(image, (x,y), radius=0, color=255, thickness=5)


def prepare_road_lanes(image, use_region_of_interest=True):
    directions = None
    if use_region_of_interest:
        # Cropping image to a triangle which should cover the road lane
        cropped_image = region_of_interest(image)
    else:
        # Using the whole image
        cropped_image = image
    try:
        # Getting road lane lines
        lines = hough_lines(cropped_image)

        # Reducing number of lines to 1 through average
        left_fit_average, right_fit_average = average_slope_intercept(lines)
        # Creating coordinates from averaged slopes/intercepts
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)

        # Extract directions only when both lanes were detected
        if left_line is not None and right_line is not None:
            directions = calculate_directions(image, left_line, right_line)

        # Rendering lines into an empty image
        image = display_lines(image, np.array([left_line, right_line]))
    except TypeError as err:
        # print(err)
        print("Unable to detect lines!")

    return image, directions
