"""YOLO demands strict file structure:
    # Dataset structure:
        # yolo/
        # -- train/
        # -- -- images/
        # -- -- labels/
        # -- val/
        # -- -- images/
        # -- -- labels/
        # -- config.yaml
config.yaml specifies paths to train, val and number of labels.
"""
import json
import os
import shutil
from os.path import isfile
from pathlib import Path

import click
import yaml
import lxml
mock_var = lxml.__version__


def yolo_converter(label):
    """
    Simple math formula to convert labels.
    in this particular task should be in [0,31] interval.
    :param label: tooth index integer "14"
    :return: label integer of [0, 31] interval.
    """
    a_arg, b_arg = str(label)
    return (int(a_arg) - 1) * 8 + (int(b_arg) - 1)


def yolo_rev_converter(num: int):
    """
    reverse operation
    :param num:
    :return:
    """
    a_arg = (1 + (num // 8)) * 10
    b_arg = 1 + (num % 8)
    return a_arg + b_arg


def json_to_yolo(outpath, json_annot):
    """
    Labels should be .txt files in form of: "label x_center y_center width height"
    coords should be relative to image size.
    :param json_annot: annotation json for specific image
    :param label_filename: annotation converted into yolo-format.
    :return: None.
    """
    with open(json_annot, mode='r', encoding="utf8") as json_file:
        data = json.load(json_file)
    # В поле file_name хранится имя изображения с расширением. Посл. обрезать.
    label_filename = data["file_name"].split(".")[0] + ".txt"
    label_filename = os.path.join(outpath, "labels", label_filename)
    with open(label_filename, mode='w', encoding="utf8") as file:
        for box in data["annotations"]:
            label = box["category_id"]
            label = yolo_converter(label)

            x_center = (box["bbox"][2] + box["bbox"][0]) / (2 * data["width"])
            y_center = (box["bbox"][3] + box["bbox"][1]) / (2 * data["height"])
            width = (box["bbox"][2] - box["bbox"][0]) / data["width"]
            height = (box["bbox"][3] - box["bbox"][1]) / data["height"]

            file.write(
                f"{label} {x_center:.5f} {y_center:.5f} {width:.5f} {height:.5f}\n"
            )


def create_yolo_yaml(dir_path):
    """build data.yaml file for yolo model"""
    data = {
        'names': {
            i:  yolo_rev_converter(i) for i in range(0, 32, 1)
        },
        'path': dir_path,
        'train': 'train/images',
        'val': 'val/images'
    }
    filepath = os.path.join(dir_path, 'data.yaml')
    with open(filepath, 'w', encoding="utf-8") as file:
        yaml.dump(data, file)


def build_paths(dir_yolo):
    """build directories for yolo model."""
    if not os.path.exists(dir_yolo):
        os.makedirs(dir_yolo)
    create_yolo_yaml(dir_yolo)
    # соберем цикл по необх. директориям.
    paths = [os.path.join(dir_yolo, "train"),
             os.path.join(dir_yolo, "val"),
             os.path.join(dir_yolo, "train", "images"),
             os.path.join(dir_yolo, "train", "labels"),
             os.path.join(dir_yolo, "val", "images"),
             os.path.join(dir_yolo, "val", "labels")]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


@click.command()
@click.option('--root_path', default="../../../")
@click.option('--threshold', default=0.8)
def run(root_path, threshold):
    """wrap function"""
    # Найти корень, откуда работаем. Должен передаваться как параметр?
    # Построить все директории из корневой папки
    src_path = os.path.join(root_path, "data", "detection")
    dest_path = os.path.join(root_path, "data", "yolo")
    build_paths(dest_path)
    img_files = [f for f in os.listdir(os.path.join(src_path,
                                                    "images")) if isfile(os.path.join(src_path,
                                                                                      "images",
                                                                                      f))]
    # train-test split threshold
    threshold = threshold * len(img_files)
    for i, img_file in enumerate(img_files):
        json_name = img_file.split(".")[0] + ".json"
        json_name = os.path.join(src_path, "labels", json_name)
        if i <= threshold:
            # внутри определит, в какую именно папку класть
            dest_tr_path = os.path.join(dest_path, "train")
            img_dest = os.path.join(dest_tr_path, "images", img_file)
            img_file = os.path.join(src_path, "images", img_file)
            json_to_yolo(dest_tr_path, json_name)
            shutil.copyfile(img_file, img_dest)
        else:
            # внутри определит, в какую именно папку класть
            dest_val_path = os.path.join(dest_path, "val")
            img_dest = os.path.join(dest_val_path, "images", img_file)
            img_file = os.path.join(src_path, "images", img_file)
            json_to_yolo(dest_val_path, json_name)
            shutil.copyfile(img_file, img_dest)


if __name__ == "__main__":
    run()
