from ultralytics import YOLO

yaml_path = './data/dataset/data.yaml'

model = YOLO('yolov8n.pt')

results = model.train(
    data=yaml_path,
    save_period=5,
    epochs=30,
    batch=32,
    imgsz=640,
    device='mps',
)
