from ultralytics import YOLO
import sys

if len(sys.argv) != 2:
    print("pass filepath to pytorch model to convert to ONNX")
    sys.exit(1)

print(len(sys.argv))

path = sys.argv[1]


YOLO(path).export(format="onnx", imgsz=640, opset=13, simplify=True)
