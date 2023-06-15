import os
from os.path import join

import pandas as pd
from torch import Tensor
import matplotlib.pyplot as plt
from ultralytics import YOLO

def filter_boxes(data: pd.DataFrame):
    """"
    Фильтруем дублирующие боксы
    """
    for i, row in data.iterrows():
        for j, sec_row in data.iterrows():
            box_a = row[0:4].tolist()
            box_b = sec_row[0:4].tolist()
            if (i != j) and get_iou(box_a, box_b) > 0.6:
                if row[4] > sec_row[4]:
                    data.loc[j, 4] = 0
                else:
                    data.loc[i, 4] = 0
    data.drop(data[data[4] == 0].index, inplace=True)
    return data



def get_iou(box_a, box_b):
    """
    calculating IoU metric between two boxes. Used to filter overlapping boxes.
    :param box_a:
    :param box_b:
    :return:
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # return the intersection over union value
    return iou

def plot_with_boxes(root_path, prediction, img, mode="show"):
    """
        Отрисует
        :param jaw_df:
        :return:
        """
    plt.imshow(prediction.plot())
    if mode == "show":
        plt.show()
    else:
        plt.savefig(join(root_path, "runs", "images", img))

def inference(root_path, img, yolo_m_path):
    """Рисуем по готовой модели"""
    model = YOLO(join(root_path, yolo_m_path))
    if type(img) == str:
        img = join(root_path, "data", "detection", "images",img)
    img_predict = model(img, verbose=False)[0]
    # фильтруем дублирующие боксы
    data = pd.DataFrame(img_predict.boxes.data.numpy())
    img_predict.boxes.data = Tensor(filter_boxes(data).to_numpy())
    return img_predict
    # рисуем
    #plot_with_boxes(root_path, img_predict, img, mode="save")


if __name__ == "__main__":
    inference("../../", '6.png', 'models/yolo_ext.pt')