# Self-Driving American Truck Simulator
This project aims to create a self-driving system for [American Truck Simulator](https://americantrucksimulator.com/) using [LaneNet](https://arxiv.org/abs/1807.01726) and [ResNet](https://ieeexplore.ieee.org/document/7780459) models.
# Goal
The project's goal is to create an autonomous self-driving truck within American Truck Simulator. All decisions should be made from data available to a player in the vanilla game. Therefore any mods simplifying the task are not allowed.

The truck should be capable of:
* Staying on the road
* Avoiding collisions
* Follow a pre-determined path (via in-game GPS)

The following tasks are out-of-scope *:
* Job completion automation
* Re-fueling
* Sleeping
* And any other non-driving related tasks
  * Menu automation
  * Garage management

<sup>* The list is not exhaustive</sup>

## Project Structure
The project consists of three separate modules
* Data pre-processing
    * Input is a full-screen screenshot of the game
    * Canny edge detection is performed
    * (optional) LaneNet overlays detected lanes
    * (optional) Simple lane detection is performed
      * A decision model may be bypassed if two lanes are detected by comparing centre of lanes and centre of image.
* Decision model
  * ResNet model trained on images collected from gameplay
    * Output: w, s, a, d
* Action output
    * Keys are sent back to the game with an X amount of delay to allow for keys to take effect
      * Forward and backward actions (w, s) have a longer delay than turning actions

# File Structure
| File                | Purpose                                                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------|
| DatasetCreator.py   | Collect raw images and inputs for training. There is a 5-second delay allowing for the active window to be switched to the game |
| ConvertRawImages.py | Pre-process collected training images into a format suitable for training                                                       |
| Train.py            | Train on pre-processed images                                                                                                   |
| Run.py              | Test the system in-game (once a model is loaded, press W to activate and Q to quit)                                             |

Please inspect each script before running for additional instructions.

# Training data
Download an example training dataset [here](https://drive.google.com/drive/folders/1-4YN3uoHxECsaCSYhczXoH7cTm9I1Rkq?usp=sharing).

# Dependencies
```
fastai==2.6.0
keyboard==0.13.5
numpy==1.22.3
Pillow==9.1.0
pyautogui==0.9.53
torch==1.11.0
torchvision==0.12.0
```
## Installation
Anaconda environment is recommended.

```
conda create --name SelfDrivingATS --file requirements.txt
conda activate SelfDrivingATS
```