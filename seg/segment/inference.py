from dotenv import load_dotenv
import os
import mlflow
from pathlib import Path
import sys

load_dotenv()

# to be import models.yolo inside mlflow.pytorch.load_model function
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ["MLFLOW_TRACKING_URI"] = os.getenv('MLFLOW_TRACKING_URI')


if __name__ == '__main__':
    MODEL_URI = os.getenv('LOAD_MODEL')
    model = mlflow.pytorch.load_model(MODEL_URI)
    # TODO prediction part
