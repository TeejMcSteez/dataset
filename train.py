from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov9s.pt")
    model.train(data="data.yaml", epochs=200, imgsz=1024, batch=16, patience=40)
