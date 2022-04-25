from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from lib.model.lanenet.LaneNet import LaneNet

'''
Extracting road lanes using LaneNet
'''

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: Train better models
model_path = Path("resource", "best_model.pth")
model = LaneNet()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
model.to(DEVICE)

data_transform = transforms.Compose([
    transforms.Resize((256, 512)),  # TODO: If new model is trained, experiment with different image sizes
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_test_data(image, transform):
    img = Image.fromarray(image)
    img = transform(img)
    return img


def extract_lanes(image):
    dummy_input = load_test_data(image, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy().astype('uint8') * 255
    return cv2.resize(binary_pred, (224, 224), interpolation=cv2.INTER_AREA)
