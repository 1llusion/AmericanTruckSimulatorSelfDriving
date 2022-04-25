# Pre-processing screen input

import pyautogui
import numpy as np
import cv2


def prepare_canny_image(image):
    image = cv2.cvtColor(image,
                         cv2.COLOR_BGR2GRAY)
    # TODO: Tune thresholds
    image = cv2.Canny(image, threshold1=550, threshold2=40)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Crops to bottom half of image.
    # image = image[0:190, 0:224]
    return image


def capture_screen():
    return np.array(pyautogui.screenshot())
