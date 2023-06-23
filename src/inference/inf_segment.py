import os
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from ultralytics.yolo.engine.results import Results

from src import inference
from src.inference import inf_yolo

warnings.filterwarnings(action='ignore', category=UserWarning)



def encode_image(image, device: str = 'cpu'):
    original_size = image.size
    width, height = original_size[0], original_size[1]
    max_size = np.max([width, height])
    height_pad = int((max_size - width) / 2)
    width_pad = int((max_size - height) / 2)
    padding = (height_pad, width_pad, height_pad, width_pad)

    image = transforms.functional.pad(image, padding, 0, 'constant')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    image = transform(image).to(device)
    return image, original_size


def decode_image(image: torch.Tensor, original_size: (int, int), device: str = 'cpu'):
    decode_transform = transforms.Compose([
        transforms.Resize(max(original_size)),
        transforms.CenterCrop(original_size[::-1])
    ])

    image = decode_transform(image)
    return image

def segment_image(img: Image, model):
    toPIL = transforms.ToPILImage()
    transform = transforms.Compose([transforms.PILToTensor()])
    tensor_img = transform(img)
    image, original_size = encode_image(img)
    predict_mask = decode_image(F.sigmoid(model(image.unsqueeze(0))).squeeze(0), original_size)
    return predict_mask


def inf_seg(yolo_prediction: Results, perbox = True):
    orig_img = yolo_prediction.orig_img
    model = torch.load('./models/efficientnet-b0-best.pth', map_location=torch.device('cpu'))
    model.eval()
    mask = torch.zeros(orig_img.shape)
    if perbox:
        for box in yolo_prediction.boxes.xyxy:
            min_x = int(box[0].item())
            min_y = int(box[1].item())
            max_x = int(box[2].item())
            max_y = int(box[3].item())
            box_img = orig_img[min_y:max_y,min_x:max_x,  1]
            segmented = segment_image(img=Image.fromarray(box_img), model=model).detach()
            mask[min_y:max_y:, min_x:max_x, 0] = segmented.squeeze()
        mask = mask.transpose(0, 2).transpose(1, 2)
    else:
        mask = segment_image(img=Image.fromarray(orig_img), model=model).detach()
    return mask

if __name__ == "__main__":
    root_path = "../../"
    img = os.path.join(root_path, "data", "detection", "images", '6.png')
    res = inference("../../", img, 'models/yolo_ext.pt')
    mask = inf_seg(res)
    toPIL = transforms.ToPILImage()
    orig = Image.open(img)
    plt.imshow(orig)
    plt.imshow(toPIL(mask), alpha=0.4)
    plt.show()