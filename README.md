# myocr
Welcome to myocr! Let's grade your answer sheets.

## Setup

```
pip install myocr
```

## Forword
Please go to [notebooks](notebooks) and run demo.ipynb
1. the demo_input.pdf in extracted to pages
2. each page is processed by image registration
3. all bounding boxes are classified to 3 classes (miss, check and cancel)


<img src="assets/image_registration.jpg" width="50%">

<img src="assets/bbox_classification.jpg" width="50%">

## Train

The demo notebook provide retrain process.
1. you need to run Label Studio before retaining.

```
docker-compose up
```
2. create task in the notebook
3. your label tasks are in http://localhost:8080/

<img src="assets/label_studio.png" width="50%">