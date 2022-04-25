import glob
import os
from pathlib import Path

import cv2
from PIL import Image

from lib.PreProcessPipeline import prepare_image

'''
Runs image pre-processing on raw training images.

Add raw images to folder 'Images_raw' or change 'FROM_DIR' to the correct folder.
Images will be written to 'Images_processed' folder or change 'TO_DIR' to the desired location.

Make sure each image is jpeg in a subdirectory labeled with the correct action.
Example:
    w/01.jpg
    s/02.jpg
    a/03.jpg
    d/04.jpg
'''
FROM_DIR = "Images_raw"
TO_DIR = "Images_processed"


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


if __name__ == '__main__':
    for image in glob.glob(FROM_DIR + '/**/*.jpg'):
        image = Path(image)
        image_name = image.name
        image_parent = image.parent.name

        to_location = Path(TO_DIR, image_parent, image_name)
        raw_image = cv2.imread(str(image))
        processed_image, _ = prepare_image(raw_image)
        if not os.path.exists(to_location.parent):
            os.makedirs(to_location.parent)

        cv2.imwrite(str(to_location), processed_image)
        # Uncomment if you want to see images being processed
        #cv2.imshow('', processed_image)
        #cv2.waitKey(1)
        print("Finished processing", image_name)
    cv2.destroyAllWindows()
    print('Done!')
