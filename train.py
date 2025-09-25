from ultralytics import YOLO
import torch

yaml_path = './data/dataset/data.yaml'

model = YOLO('yolov8n.pt')

if torch.backends.mps.is_available():
    mps_device = torch.device('mps')
    x = torch.ones(1, device=mps_device)
    print(x)
    results = model.train(
        data=yaml_path,
        save_period=5,
        epochs=30,
        batch=32,
        imgsz=640,
        device='mps',
    )
else:
    print('MPS device not found.')
    results = model.train(
        data=yaml_path,
        save_period=5,
        epochs=30,
        batch=32,
        imgsz=640,
    )
