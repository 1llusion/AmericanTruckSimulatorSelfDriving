from fastai.vision.all import *
import keyboard
import numpy as np
import cv2
import pyautogui
import time

from lib.Image.ImagePreProcessing import capture_screen
from lib.PreProcessPipeline import prepare_image


# Send input to game
def do_action(action):
    print(action)
    keyboard.press(action)
    # Extra time is needed when going forward/backwards
    if action == 'w' or action == 's':
        time.sleep(0.5)
    else:
        # Longer time allows for tight corners to be cleared, but decreases precision
        time.sleep(0.01)
    keyboard.release(action)


if __name__ == '__main__':
    print("Loading Model...")
    model = load_learner("Images_line_test/export.pkl", cpu=False)
    print("Model loaded!")
    print("Press W to start!")
    keyboard.wait("w")
    print("Press Q to stop!")

    while not keyboard.is_pressed("q"):
        image = capture_screen()
        image, action = prepare_image(image)
        # If lanes were not sufficiently detected, use model to predict which action should be taken
        if action is None:
            action = model.predict(image)[0]
        do_action(action)
        cv2.imshow('Screenshot', cv2.resize(image, (800, 400)))
        cv2.waitKey(1)
    cv2.destroyAllWindows()
