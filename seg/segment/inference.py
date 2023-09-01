from dotenv import load_dotenv
import os
import mlflow
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

load_dotenv()

# to be import models.yolo inside mlflow.pytorch.load_model function
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from export import export_formats
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    check_suffix,
    check_file,
    increment_path,
    check_img_size,
    check_imshow
)
from utils.torch_utils import select_device

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')


class DetectMultiBackend(nn.Module):
    """
        set structture of model that loaded from mlflow
        as same as DetectMultiBackend in predice.py
        ONLY .pt
    """
    def __init__(self,
                 raw_model,
                 pt=True,
                 device=torch.device('cpu'),
                 dnn=False,
                 data=None,
                 fp16=False,
                 fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        super().__init__()
        w = '.pt' if pt else None  # force to set pt to True
        # TODO: 
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs
         ) = self._model_type(w)  # get backend


    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs

if __name__ == '__main__':
    source = 'datasets/horoscope/test'
    imgsz = (640, 640)
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt = True
    nosave = False
    project = ROOT / 'runs/predict-seg'
    name = 'exp'
    exist_ok = False  # existing project/name ok, do not increment
    half = False
    pt = True  # is .pt file

    # source
    source = str(source)
    # save inference images
    save_img = not nosave and not source.endswith('.txt')
    # check source is an image / dir?
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # check source comes from url?
    is_url = source.lower().startswith(('rtsp://',
                                        'rtmp://',
                                        'http://',
                                        'https://'))
    # check source is webcam?
    webcam = (source.isnumeric()
              or
              source.endswith('.txt')
              or
              (is_url and not is_file))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories (output)
    # increment run
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # make dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)

    # Load model
    device = select_device(device)
    MODEL_URI = os.getenv('LOAD_MODEL')
    model = mlflow.pytorch.load_model(MODEL_URI)
    # ref from models.common import DetectMultiBackend L404
    stride = max(int(model.stride.max()), 32)  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    fp16 = half
    model.half() if fp16 else model.float()
    segmentation_model = type(model.model[-1]).__name__ in ['Segment', 'ISegment', 'IRSegment']
    model.stride = stride
    model.names = names
    model.pt = pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference

