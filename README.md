# yolov7

Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

This implimentation is based on [yolov5](https://github.com/ultralytics/yolov5).

## Object detection

[code](./det)

## Instance segmentation

[code](./seg)

Train:
```
! cd seg
python segment/train.py --batch-size 8 \
 --img-size 640 \
 --epochs 10 \
 --data data/horoscope.yml \
 --weights data/weights/yolov7-seg.pt \
 --device 0 \
 --name model
```

Detect:
```
! cd seg
python segment/predict.py \
--weights runs/train-seg/model/weights/best.pt \
--conf 0.25 \
--data data/horoscope.yml \
--source ../datasets/horoscope/test \
--name evaluate_model
```

## Image classification

[code](./det)
