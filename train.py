import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import YOLO

# test git

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-bifpn.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=10,
                workers=4,
                device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )