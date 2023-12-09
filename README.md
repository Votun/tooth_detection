# DentalVision

Развёрнутый проект доступен по адресу http://213.171.9.105:8050

### Стек технологий 

![](https://img.shields.io/badge/Python-3.10-black?style=for-the-badge&logo=python) 
![](https://img.shields.io/badge/Pandas-2.0.2-black?style=for-the-badge&logo=pandas)
![](https://img.shields.io/badge/MLflow-2.4.1-black?style=for-the-badge&logo=mlflow)
![](https://img.shields.io/badge/Flask-2.2.5-black?style=for-the-badge&logo=flask)
![](https://img.shields.io/badge/Docker-black?style=for-the-badge&logo=docker)
![](https://img.shields.io/badge/PyTorch-2.0.1-black?style=for-the-badge&logo=pytorch)
![](https://img.shields.io/badge/ultralytics-8.0.119-black?style=for-the-badge&logo=ultralytics)
![](https://img.shields.io/badge/dvc-3.0-black?style=for-the-badge&logo=dvc)

# Платформа для сегментации зубов на панорамных снимках челлюсти
Идея:
Сейчас большинство данных о пациентах в стоматологиях - неоцифрованны, что значительно усложняет последующее взаимодействие с этими данными, тем самым увеличивая продолжительность лечения. Платформа дает возможность автоматической разметки номеров зубов челюсти в соответствии с классификацией FDI, находить повреждения на зубах, предлагая при этом интерфейс для работы с данными.

![dfc892d7c08bca164e2765b19e585d50](https://github.com/Votun/tooth_detection/assets/16477307/9129b565-f14c-449b-a97a-d06b35fe5555)

## Цели платформы:
- С высокой точностью размечать зубы по типу и номеру (IoU + average classif. metrics).
- С высокой точностью определять поражения кариесом на снимке, метрике IoU.
- создание минимального интерфейса, позволяющего обрабатывать снимки.
- Отоборажение статистики по модели.
## Данные.
Отправной точкой стал размеченный вручную сет из 200 снимков. Разметка проводилась в vast.ai, после чего файлы разметки парсились для работы в различных моделях. Затем был найден более обширный сет [Tufts Dental Database](http://tdd.ece.tufts.edu/). Для сегментации поражений зубов были привлечен небольшой набор данных из открытого репозитория [Panoramic-Caries-Segmentation](https://github.com/Zzz512/MLUA) по статье [Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0925231223003193?via%3Dihub).

## Материалы по теме
|Date|The First and Last Authors|Title|Code|Reference
|---|---|---|---|---|
|2019|Hu Chen, Chin-Hui Lee|A deep learning approach to automatic teeth detection and numbering based on object detection in dental periapical films|None|[Article](https://www.nature.com/articles/s41598-019-40414-y)|
|April 2022|Karen Panetta,  Rahul Rajendran|Tufts Dental Database: A Multimodal Panoramic X-Ray Dataset for Benchmarking Diagnostic Systems|[Data](http://tdd.ece.tufts.edu/)|[Article](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9557804)|
|April 2023|Xianyun Wang, Fan Yang|Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation| [Code](https://github.com/Zzz512/MLUA)|[Article](https://www.sciencedirect.com/science/article/abs/pii/S0925231223003193?via%3Dihub)|
|April 2022|Mrinal Kanti Dhar, Mou Deb|S-R2F2U-Net: A single-stage model for teeth segmentation|[Code](https://github.com/mrinal054/teethSeg_sr2f2u-net)|[Article](https://arxiv.org/abs/2204.02939)|

## Подход.
#### 1. Детекция и классификация зубов.
На основе статьи "A deep learning approach to automatic teeth detection..." было обучено несколько моделей на архитектуре Faster RCNN и YOLO, с дальнейшей постобработкой. Постобработка включала удаление дублирующих боксов, правка ошибок классификации. Тем не менее, после обучения YOLOv8n на расширенном сете снимков удалось добиться достаточно высокой точности без привлечения сложных методов постобработки, предложенных в статье.
#### 2. Сегментация.
Модель Unet обученная на сете Panoramic Caries Segmentation. Были проведены эксперименты с разными конфигурациями, предполагалась работа как с вырезанными фрагментами снимков, так и полными.
#### 3. UI.
Работа не предполагала создания полновесного сервиса, поэтому был написан MVP на Flask и Dash, позвоялющий принимать снимки и визуализировать результат их обработки.

 
