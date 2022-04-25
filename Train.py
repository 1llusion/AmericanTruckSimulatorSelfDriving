from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders

'''
Train a new model on processed training data.
Change 'PATH' if training images are not located in 'Images_processed' folder.

Make sure each image is located in a sub-folder named with the correct label.
Example:
    Images_processed/w/01.jpg
    Images_processed/s/02.jpg
    Images_processed/a/03.jpg
    Images_processed/d/04.jpg
'''

PATH = "Images_processed"


def get_label(x):
    return x.parent.name


files = get_image_files(PATH)
print("Loaded {} images.".format(len(files)))

dataLoader = ImageDataLoaders.from_path_func(PATH, files, get_label, bs=40, num_workers=0)
print("Loading model...")
model = cnn_learner(dataLoader, resnet18, metrics=error_rate)
print("Model loaded!")
print("Tuning model...")
model.fine_tune(10, base_lr=1.0e-02)
print("Model tuned!")
print("Exporting model...")
model.export()
print("Model exported!")
print("Done!")
