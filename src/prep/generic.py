# Переименовать файлы. Свести кодирование боксов к одному виду.
# В result папки images и labels.
# Боксы заданы по координатам верхнего левого и правого нижнего углов. Остальное - дополнительно.
import os
import json
import shutil
import sys
import xml.etree.ElementTree as ET

import click as click
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
import torch
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def coco_converter(coco_cat_id:int) -> int:
    """
    in coco dataset teeth are indexed 1-32 according to "clock round"
    i.e. 1 == 18, 16 == 28, 32 == 38.
    :param coco_cat_id: index of the tooth
    :return: label acc. to dental notation.
    """
    logger.info("Конвертация меток coco в формат 18/48...")
    dec = ((coco_cat_id - 1) // 8) + 1
    if (dec % 2) == 1:
        digit = 8 - ((coco_cat_id - 1) % 8)
    else:
        digit = coco_cat_id % 8
    return 10*dec + digit

@click.group()
def run():
    pass

@run.command('from_raw')
@click.option('-d', '--datapath', default='raw', help="Path to data folder.")
@click.argument('ann_file')
@click.option('-o', '--output', default='detection')
def from_raw(datapath: str, ann_file: str, output: str):
    """
    копирует изображения из папки raw в папку detection/images,
    разбивает разметку на отдельные файлы в папке detection/labels.
    :param filepath: path to directory.
    :return:
    """
    # Создать необходимые директории и проверить корректность
    img_out, labels_out = check_paths(datapath, output)
    filepath = os.path.join(datapath, ann_file)
    if not os.path.exists(filepath):
        logger.error("Файл аннотации не найден.")
    tree = ET.parse(filepath)
    img_object = {}
    counter = 0
    for xml_object in tree.findall("image"):
        counter += 1
        box_list = []
        for box in xml_object.findall("box"):
            ann_object = {"bbox": [
                float(box.attrib["xtl"]),
                float(box.attrib["ytl"]),
                float(box.attrib["xbr"]),
                float(box.attrib["ybr"])
            ], "category_id": box.attrib["label"]}
            box_list.append(ann_object)

        img_object["annotations"] = box_list
        img_object["file_name"] = xml_object.attrib["name"]
        img_object["height"] = float(xml_object.attrib["height"])
        img_object["width"] = float(xml_object.attrib["width"])
        with open(os.path.join(labels_out, f"{xml_object.attrib['name'].split('.')[0]}.json"), 'w') as lbl_file:
            json.dump(img_object, lbl_file)
        img_src = os.path.join(datapath, "images", img_object["file_name"])
        img_dest = os.path.join(img_out, img_object["file_name"])
        shutil.copyfile(img_src, img_dest)
    logger.info(f"Обработано {counter} снимков.")


@run.command('from_ext')
@click.option('-d', '--datapath', default='', help="Path to data folder.")
@click.option('--ann_file', default='coco_annotations.json', help="Specify name of annotation file.")
@click.option('-o', '--output', default='detection')
def from_ext(datapath, ann_file, output):
    """
    in external are: images folder and some annot.json
    :param datapath: path to directory.
    :return:
    """
    # Создать необходимые директории и проверить корректность
    img_out, labels_out = check_paths(datapath, output)

    filepath = os.path.join(datapath, ann_file)
    if not os.path.exists(filepath):
        logger.error("Файл аннотации не найден.")
    with open(filepath, 'r') as f:
        data = json.load(f)
    for img_object in data:
        # Все ненужные поля обрезаны
        img_object.pop("dataset", None)
        res_name = img_object.pop("image_id", None)
        img_object.pop("num_teeth", None)
        img_object.pop("original_filename", None)
        # Список боксов. Метки нужно перекодировать,
        box_list = []
        for ann_object in img_object["annotations"]:
            ann_object["category_id"] = str(coco_converter(int(ann_object["category_id"])))
            ann_object.pop("bbox_mode", None)
            box = ann_object["bbox"]
            box[2] = box[0] + box[2]
            box[3] = box[1] + box[3]
            ann_object["bbox"] = box
            box_list.append(ann_object)

        img_object["annotations"] = box_list
        with open(os.path.join(labels_out, res_name + ".json"), 'w') as lbl_file:
            json.dump(img_object, lbl_file)
        img_src = os.path.join(datapath, "images", img_object["file_name"])
        img_dest = os.path.join(img_out, img_object["file_name"])
        shutil.copyfile(img_src, img_dest)
    logger.info(f"Обработано {len(data)} снимков.")


def check_paths(datapath: str, output: str):
    """
    Проверка путей и создание необходимых директорий
    """
    if not os.path.exists(datapath):
        logger.error("Source директория не найдена.")
    if not os.path.exists(output):
        os.makedirs(output)
        logger.info(f"Создана директория {output}.")
    img_out = os.path.join(output, "images")
    labels_out = os.path.join(output, "labels")
    if not os.path.exists(img_out):
        os.makedirs(img_out)
        logger.info(f"Создана директория {img_out}.")
    if not os.path.exists(labels_out):
        os.makedirs(labels_out)
        logger.info(f"Создана директория {labels_out}.")
    return img_out, labels_out


def plot_image_boxes(image, datapath: str, boxes=True):
    """plot image with/without boxes.
       Check annotation
    """
    img = read_image(os.path.join(datapath, "images", image), mode=ImageReadMode.RGB)
    if boxes:
        img_data = {}
        with open(os.path.join(datapath, "labels", image.split('.')[0]+".json"), 'r') as file:
            img_data = json.load(file)
        annotations = img_data["annotations"]
        boxes_tensor = torch.empty(0, 4)
        labels = []
        for ann_object in annotations:
            row = torch.Tensor([float(x) for x in ann_object["bbox"]]).unsqueeze(0)
            boxes_tensor = torch.cat((boxes_tensor, row), dim=0)
            labels.append(ann_object["category_id"])
        # boxes_tensor = torchvision.ops.box_convert(boxes_tensor, 'xxyy', 'xyxy')
        img = draw_bounding_boxes(img, boxes_tensor, labels)
    img = img.permute(1,2,0)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(img)
    plt.show()


if __name__ == "__main__":
    run()
