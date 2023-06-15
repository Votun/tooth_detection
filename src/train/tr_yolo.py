"""
Train module. Includes all methods for different detection models' training
"""
import os
from pathlib import Path

import click
import yaml

from ultralytics import YOLO


@click.command()
@click.option('--data_path', default="data/detection/yolo", help='Path to data.')
@click.option('--output', default="runs/detection", help='Where data would be saved.')
@click.option('--max_epochs', default=1, help='Number of epochs.')
@click.option('--exp_id', default=0)
def train_yolo(data_path,
               output,
               exp_id,
               model_type="yolov8n.pt",
               data_config="data.yaml",
               max_epochs=1):
    """
    A shell for yolo train process.
    YOLO python api needs some adjustments, specifically with paths processing.
    Absolute paths are needed. Parameters of train process itself are to be added.
    :param data_config: a name of data configuration yaml. Should lie in data directory.
    :param model_type: type of yolo model. 8nano as default
    :param data_path: Path to data folder. Full paths are needed.
    :param result_path: Path to experiment results folder.
    :param max_epochs: number of epochs
    :return:
    """
    root_path = Path(__file__).parent.parent.parent
    data_path = os.path.join(root_path, data_path)
    result_path = os.path.join(root_path, output)

    os.environ["WANDB_DISABLED"] = "true"
    model = YOLO(model_type)
    # Use the model
    filepath = os.path.join(data_path, data_config)
    # Path in data config must correspond to data_path
    with open(filepath, 'r', encoding="utf-8") as config_file:
        doc = yaml.load(config_file, Loader=yaml.FullLoader)
    doc['path'] = data_path
    with open(filepath, 'w', encoding="utf-8") as config_file:
        yaml.dump(doc, config_file)
    model.train(data=filepath, epochs=max_epochs, project=result_path)


if __name__ == "__main__":
    # path = "data/detection/yolo"
    # result_path = "runs"
    train_yolo()
