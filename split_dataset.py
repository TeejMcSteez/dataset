"""
Split a CVAT 'YOLO 1.1' / 'Ultralytics YOLO' export into train/val.

Input:  a directory (default: obj_train_data) containing image files and
        their matching .txt label files side by side (CVAT's layout).
Output: images/{train,val} and labels/{train,val} created next to this script.

Each image is kept paired with its own .txt (same basename) and the PAIR is
assigned to a split together, so a label never ends up in a different split
than its image. Default split is 80% train / 20% val.
"""

import random
import shutil
from pathlib import Path

# ---- settings ----------------------------------------------------
SRC       = Path("obj_train_data")   # CVAT export folder (images + .txt together)
OUT       = Path(".")                # where images/ and labels/ get created
VAL_FRAC  = 0.20                      # fraction sent to validation
SEED      = 42                        # fixed seed => same split every run
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}

# Scene-leakage guard: if your frames from one clip share a name prefix
# (e.g. porch_0012.jpg, porch_0013.jpg), set this True to keep whole groups
# on one side of the split so near-identical frames don't leak into val.
# Only useful if your filenames actually encode the clip in the prefix.
GROUP_BY_PREFIX   = False
PREFIX_SPLIT_CHAR = "_"              # "porch_0012" -> group "porch"
# ------------------------------------------------------------------


def main():
    random.seed(SEED)

    if not SRC.is_dir():
        raise SystemExit(f"Source folder not found: {SRC.resolve()}")

    images = sorted(p for p in SRC.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {SRC.resolve()}")

    # Pair every image with its label. A missing .txt means "no annotations"
    # (Ultralytics treats that as an empty/background image).
    pairs, missing = [], []
    for img in images:
        txt = img.with_suffix(".txt")
        if txt.exists():
            pairs.append((img, txt))
        else:
            pairs.append((img, None))
            missing.append(img.name)

    # Decide the split.
    if GROUP_BY_PREFIX:
        groups = {}
        for pair in pairs:
            key = pair[0].stem.split(PREFIX_SPLIT_CHAR)[0]
            groups.setdefault(key, []).append(pair)
        keys = list(groups)
        random.shuffle(keys)
        n_val_keys = max(1, round(len(keys) * VAL_FRAC))
        val_keys = set(keys[:n_val_keys])
        val   = [p for k in val_keys for p in groups[k]]
        train = [p for k in keys if k not in val_keys for p in groups[k]]
    else:
        random.shuffle(pairs)
        n_val = max(1, round(len(pairs) * VAL_FRAC))
        val, train = pairs[:n_val], pairs[n_val:]

    # Create output folders.
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    def copy_split(split_pairs, name):
        for img, txt in split_pairs:
            shutil.copy2(img, OUT / "images" / name / img.name)
            if txt is not None:
                shutil.copy2(txt, OUT / "labels" / name / txt.name)

    copy_split(train, "train")
    copy_split(val, "val")

    # Report.
    print(f"Total pairs : {len(pairs)}")
    print(f"  train     : {len(train):>5}  -> images/train, labels/train")
    print(f"  val       : {len(val):>5}  -> images/val,   labels/val")
    if missing:
        print(f"\n{len(missing)} image(s) had NO .txt (copied as empty/background):")
        for n in missing[:10]:
            print(f"    {n}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
