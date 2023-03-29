from ultralytics import YOLO
import torch

# Set the device
dtype = torch.float
device = torch.device("mps")


# # yolo model creation
model = YOLO("yolo-weights/yolov8m.pt")
model.train(data="coco128.yaml", imgsz=320, batch=4, epochs=100, workers=0)

# prediction
model = YOLO("best.pt")
results = model.predict("videos/3.mp4", save=True, conf=0.5, show=True)


