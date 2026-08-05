"""
Thin background (unlabeled) frames from a YOLO training split WITHOUT
deleting anything. Moved files go to a holding folder so the dataset stays
clean and every move is reversible (just move them back).

Strategy — "keep the hard negatives, drop the boring duplicates":
  1. Find background images (label .txt is empty or missing).
  2. Perceptual-hash them and drop NEAR-DUPLICATES first, keeping one
     representative per cluster. Visually distinct empties (headlights,
     moving trees, night/IR, shadows) survive; repeated identical empty
     scenes collapse to a single kept frame.
  3. If distinct backgrounds still exceed the target ratio, randomly move
     the extras until the target is hit.

Only the TRAIN split is thinned. Leave val alone — it should mirror the
real scene distribution so your validation metrics stay honest.

Run from your dataset root (where images/ and labels/ live).
Requires: pip install pillow imagehash
"""

import random
import shutil
from pathlib import Path

try:
    from PIL import Image
    import imagehash
except ImportError:
    raise SystemExit("Missing deps. Run: pip install pillow imagehash")

# ---- settings ----------------------------------------------------
IMAGES_DIR = Path("images/train")
LABELS_DIR = Path("labels/train")
HELD_DIR   = Path("backgrounds_held")   # moved (not deleted) files land here
TARGET_BG_RATIO = 0.25    # desired backgrounds as a fraction of the final set
PHASH_THRESH    = 6       # Hamming distance; lower = only near-identical dropped
SEED    = 42
DRY_RUN = True            # True = report only, move nothing. Flip to False to act.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# ------------------------------------------------------------------


def is_background(img: Path) -> bool:
    txt = LABELS_DIR / (img.stem + ".txt")
    # background = label file missing OR present but empty (no boxes)
    return (not txt.exists()) or txt.stat().st_size == 0


def main():
    random.seed(SEED)
    images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images in {IMAGES_DIR.resolve()}")

    backgrounds = [p for p in images if is_background(p)]
    labeled_count = len(images) - len(backgrounds)
    print(f"Images total      : {len(images)}")
    print(f"  labeled         : {labeled_count}")
    print(f"  backgrounds     : {len(backgrounds)}  "
          f"({len(backgrounds) / len(images):.0%})")

    # target background count for the desired ratio:  bg = r*labeled/(1-r)
    r = TARGET_BG_RATIO
    target_bg = round(r * labeled_count / (1 - r))
    print(f"Target backgrounds: {target_bg}  (~{r:.0%} of final set)")

    if len(backgrounds) <= target_bg:
        print("Already at or below target — nothing to do.")
        return

    # --- stage 1: hash and drop near-duplicates ----------------------
    print("\nHashing backgrounds (this can take a moment)...")
    hashed = []
    for p in backgrounds:
        try:
            with Image.open(p) as im:
                hashed.append((p, imagehash.phash(im)))
        except Exception as e:
            print(f"  skip unreadable {p.name}: {e}")

    random.shuffle(hashed)   # so the kept representative isn't order-biased
    kept, kept_hashes, dup_moves = [], [], []
    for p, h in hashed:
        if any(h - kh <= PHASH_THRESH for kh in kept_hashes):
            dup_moves.append(p)          # near-identical to something already kept
        else:
            kept.append(p)
            kept_hashes.append(h)

    print(f"  distinct kept   : {len(kept)}")
    print(f"  near-dupes moved: {len(dup_moves)}")

    # --- stage 2: if still over target, move random distinct extras ---
    extra_moves = []
    if len(kept) > target_bg:
        random.shuffle(kept)
        extra_moves = kept[target_bg:]
        kept = kept[:target_bg]
        print(f"  extra distinct moved to hit target: {len(extra_moves)}")

    to_move = dup_moves + extra_moves
    final_bg = len(backgrounds) - len(to_move)
    final_ratio = final_bg / (final_bg + labeled_count)
    print(f"\nWill move {len(to_move)} backgrounds -> {HELD_DIR}/")
    print(f"Resulting backgrounds: {final_bg}  ({final_ratio:.0%} of final set)")

    if DRY_RUN:
        print("\nDRY_RUN is True — nothing moved. Set DRY_RUN=False to apply.")
        return

    # --- move image + its label together, preserving structure -------
    (HELD_DIR / "images").mkdir(parents=True, exist_ok=True)
    (HELD_DIR / "labels").mkdir(parents=True, exist_ok=True)
    for img in to_move:
        shutil.move(str(img), HELD_DIR / "images" / img.name)
        txt = LABELS_DIR / (img.stem + ".txt")
        if txt.exists():
            shutil.move(str(txt), HELD_DIR / "labels" / txt.name)
    print(f"\nDone. Moved {len(to_move)} pairs to {HELD_DIR}/ (recoverable).")


if __name__ == "__main__":
    main()
