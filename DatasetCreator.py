import os
import time

import cv2
import keyboard

from lib.Image.ImagePreProcessing import capture_screen

'''
Capture raw training images.
Saves images to 'Images_raw' folder.
'''


def save_data(image, key):
    path = "Images_raw/{}".format(key)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = "{}/{}.jpg".format(path, time.time())
    cv2.imwrite(filename, image)


if __name__ == '__main__':
    for i in range(5, 0, -1):
        print("Starting in {} seconds!".format(i))
        time.sleep(1)
    print("Capturing!")
    while not keyboard.is_pressed("q"):
        image = capture_screen()
        key = keyboard.read_key()
        save_data(image, key)
        cv2.imshow('Screenshot', cv2.resize(image, (800, 400)))
        cv2.waitKey(100)
    cv2.destroyAllWindows()
