"""
Compare two YOLO models side by side on UNLABELED footage (a video file or a
folder of images). Runs both models on every frame, draws their detections in
different colors, writes an annotated output, and prints summary stats so you
can see where the old and new models agree and disagree.

This is a QUALITATIVE generalization check on fresh footage. For hard numbers
(mAP / precision / recall) run each model against your labeled val split:
    yolo val model=old.onnx data=data.yaml imgsz=320
    yolo val model=new.pt   data=data.yaml imgsz=640

Usage:
    python compare_models.py --old old.onnx --new new.pt \
        --source clip.mp4 --out comparison.mp4 --imgsz 640 --conf 0.25

    # or a folder of images (writes annotated images into --out):
    python compare_models.py --old old.onnx --new new.pt \
        --source ./new_clips --out ./compared --imgsz 640

Requires: pip install ultralytics opencv-python
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

OLD_COLOR = (0, 0, 255)    # red   (BGR) -> OLD model
NEW_COLOR = (0, 255, 0)    # green (BGR) -> NEW model
VID_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def draw_and_count(frame, results, color, names, label_top):
    """Draw boxes for one model's results; return how many it found."""
    n = 0
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf, cls = float(b.conf[0]), int(b.cls[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{names.get(cls, cls)} {conf:.2f}"
            ty = y1 - 6 if label_top else y2 + 16
            cv2.putText(frame, tag, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
            n += 1
    return n


def accumulate(stats, results):
    hit = False
    for r in results:
        for b in r.boxes:
            stats["total"] += 1
            stats["conf_sum"] += float(b.conf[0])
            stats["per_class"][int(b.cls[0])] += 1
            hit = True
    if hit:
        stats["frames_hit"] += 1


def report(stats, name, model_path, names):
    print(f"\n=== {name}  ({model_path}) ===")
    print(f"  total detections : {stats['total']}")
    print(f"  frames with >=1  : {stats['frames_hit']}")
    if stats["total"]:
        print(f"  mean confidence  : {stats['conf_sum'] / stats['total']:.3f}")
    for cls, c in sorted(stats["per_class"].items()):
        print(f"    {names.get(cls, cls)!s:>10} : {c}")


def frame_source(src: Path):
    """Yield (frame, frame_id) from a video or an image folder."""
    if src.is_file() and src.suffix.lower() in VID_EXTS:
        cap = cv2.VideoCapture(str(src))
        meta = {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 25,
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "video": True,
        }
        def gen():
            i = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f, f"{i:06d}"
                i += 1
            cap.release()
        return gen(), meta
    elif src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        def gen():
            for p in imgs:
                yield cv2.imread(str(p)), p.name
        return gen(), {"video": False}
    else:
        raise SystemExit(f"--source must be a video file or an image folder: {src}")


def run(args):
    old = YOLO(args.old)
    new = YOLO(args.new)
    names = new.names  # assumes both models share the same label set

    frames, meta = frame_source(Path(args.source))

    writer = None
    if meta["video"]:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 meta["fps"], (meta["w"], meta["h"]))
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    stats = {m: {"total": 0, "frames_hit": 0, "conf_sum": 0.0,
                 "per_class": defaultdict(int)} for m in ("old", "new")}
    disagree = 0
    n_frames = 0

    for frame, fid in frames:
        if frame is None:
            continue
        n_frames += 1
        ro = old(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        rn = new(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

        no = draw_and_count(frame, ro, OLD_COLOR, names, label_top=True)
        nn = draw_and_count(frame, rn, NEW_COLOR, names, label_top=False)
        accumulate(stats["old"], ro)
        accumulate(stats["new"], rn)
        if no != nn:
            disagree += 1

        if writer:
            writer.write(frame)
        else:
            cv2.imwrite(str(Path(args.out) / f"cmp_{fid}"), frame)

    if writer:
        writer.release()

    print(f"\nProcessed {n_frames} frames.")
    print(f"RED = OLD ({args.old})    GREEN = NEW ({args.new})")
    report(stats["old"], "OLD", args.old, names)
    report(stats["new"], "NEW", args.new, names)
    print(f"\nFrames where the two models' detection COUNT differed: "
          f"{disagree} / {n_frames}")
    print(f"Annotated output: {args.out}")
    print("\nNOTE: without labels this shows AGREEMENT, not accuracy. "
          "Eyeball the disagreements, and use `yolo val` on your val split "
          "for real mAP/precision/recall numbers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="old model path (.pt or .onnx)")
    ap.add_argument("--new", required=True, help="new model path (.pt or .onnx)")
    ap.add_argument("--source", required=True, help="video file OR folder of images")
    ap.add_argument("--out", default="comparison.mp4",
                    help="output video path, or output folder if source is a folder")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size; must match each model's export/train size")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    run(args)
