from lib.Image.ImagePreProcessing import prepare_canny_image
from lib.Image.RoadLaneDetection import extract_lanes
from lib.Image.RoadLaneProcessing import *

'''
Image pre-processing pipeline used during conversion of raw images to training data and running model prediction.


use_lanenet = Use LaneNet (https://github.com/klintan/pytorch-lanenet) to detect lanes?
draw_lines = Detect lines from image? Additionally returns direction, which can be directly used as an action
region_of_interest = Should the image be cropped to a region where lanes should be located?
                     This greatly reduces noise in the image. However can cause lanes to not be visible in edge cases

LaneNet can be used together with draw_image to get more accurate directions.
'''


def get_line_image_and_directions(image, use_lanenet=False, draw_lines=True):
    direction = None
    if use_lanenet:
        image = extract_lanes(image)
    if draw_lines:
        image, direction = prepare_road_lanes(image, use_region_of_interest=False)
    return image, direction


def prepare_image(image):
    canny_image = prepare_canny_image(image)
    line_image, directions = get_line_image_and_directions(image, use_lanenet=True, draw_lines=False)
    # Combining empty image with lines with canny image
    combined_image = cv2.addWeighted(canny_image, 0.8, line_image, 1, 1)
    return combined_image, directions
