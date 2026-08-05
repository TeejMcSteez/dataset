import PIL.Image
import cvat_sdk.auto_annotation as cvataa
from ultralytics import YOLO

_model = YOLO("yolo26n.pt")   # or your trained best.pt

# These names MUST match the label names in your CVAT task exactly.
spec = cvataa.DetectionFunctionSpec(labels=[
    cvataa.label_spec("person", 0),
    cvataa.label_spec("dog", 1),
])

def detect(context, image: PIL.Image.Image):
    shapes = []
    for r in _model.predict(source=image, conf=0.25, classes=[0, 16], verbose=False):
        for b in r.boxes:
            # remap COCO ids -> your task ids: person 0->0, dog 16->1
            coco = int(b.cls[0])
            label_id = 0 if coco == 0 else 1
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            shapes.append(cvataa.rectangle(label_id, [x1, y1, x2, y2]))
    return shapes
