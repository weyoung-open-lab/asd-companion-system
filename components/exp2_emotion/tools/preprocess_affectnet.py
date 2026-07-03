"""
AffectNet Optimized Preprocessor
==================================
Combines face cropping + class balancing + multi-threading in one script.

What it does:
1. Reads training.csv and validation.csv
2. For each valid expression (0-7), determines how many to keep (balanced)
3. Crops face region using CSV coordinates
4. Resizes to 224x224 (ready for ResNet-50)
5. Saves into class-named folders (0_Neutral, 1_Happy, etc.)
6. Uses multi-threading for speed

Output structure:
  AffectNet_Processed/
    train/
      0_Neutral/    img001.jpg, img002.jpg, ...
      1_Happy/      ...
      ...
      7_Contempt/   ...
    val/
      0_Neutral/    ...
      ...

Usage: python preprocess_affectnet.py
Estimated time: 15-30 minutes (depending on disk speed)
"""

import os
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, UnidentifiedImageError
from collections import defaultdict
import time

# ============================================================
# UPDATE THESE PATHS
# ============================================================
IMAGE_ROOT = r"D:\BaiduNetdiskDownload\Manually_Annotated_Images"
TRAIN_CSV = r"D:\BaiduNetdiskDownload\Manually_Annotated_file_lists\training.csv"
VALID_CSV = r"D:\BaiduNetdiskDownload\Manually_Annotated_file_lists\validation.csv"
OUTPUT_ROOT = r"D:\BaiduNetdiskDownload\AffectNet_Processed"

# Balancing config
MAX_PER_CLASS_TRAIN = 15000   # Cap large classes at 15000
MAX_PER_CLASS_VAL = 500       # Keep validation balanced (already 500 each)
TARGET_SIZE = (224, 224)      # Resize for ResNet-50
NUM_WORKERS = 8               # Number of threads
RANDOM_SEED = 42

EXPRESSION_MAP = {
    0: "0_Neutral",
    1: "1_Happy",
    2: "2_Sad",
    3: "3_Surprise",
    4: "4_Fear",
    5: "5_Disgust",
    6: "6_Anger",
    7: "7_Contempt",
}


# ============================================================
# Step 1: Read CSV and select balanced subset
# ============================================================
def load_and_balance_csv(csv_path, max_per_class, label=""):
    """Load CSV, filter valid expressions, balance by downsampling."""
    print(f"\n  Loading {label} CSV: {csv_path}")

    rows_by_class = defaultdict(list)
    skipped = 0

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            try:
                expression = int(row[6])
                if expression not in range(8):
                    skipped += 1
                    continue

                # Check for NULL coordinates
                if 'NULL' in row[1:5]:
                    skipped += 1
                    continue

                face_x = int(float(row[1]))
                face_y = int(float(row[2]))
                face_w = int(float(row[3]))
                face_h = int(float(row[4]))

                rows_by_class[expression].append({
                    "path": row[0],
                    "face_x": face_x,
                    "face_y": face_y,
                    "face_w": face_w,
                    "face_h": face_h,
                    "expression": expression,
                })
            except (ValueError, IndexError):
                skipped += 1
                continue

    # Print original distribution
    print(f"  Skipped {skipped} invalid rows")
    print(f"\n  {'Expression':<15} {'Original':>8} {'After Balance':>14}")
    print(f"  {'-'*40}")

    total_original = 0
    total_balanced = 0
    balanced_rows = []

    random.seed(RANDOM_SEED)

    for expr_id in range(8):
        name = EXPRESSION_MAP[expr_id]
        original = rows_by_class[expr_id]
        n_original = len(original)
        total_original += n_original

        if n_original <= max_per_class:
            selected = original
        else:
            selected = random.sample(original, max_per_class)

        n_selected = len(selected)
        total_balanced += n_selected
        balanced_rows.extend(selected)

        marker = "" if n_original <= max_per_class else " (capped)"
        print(f"  {name:<15} {n_original:>8} {n_selected:>14}{marker}")

    print(f"  {'TOTAL':<15} {total_original:>8} {total_balanced:>14}")

    if total_original > 0:
        ratio_before = max(len(v) for v in rows_by_class.values()) / max(1, min(len(v) for v in rows_by_class.values() if len(v) > 0))
        counts_after = defaultdict(int)
        for r in balanced_rows:
            counts_after[r["expression"]] += 1
        ratio_after = max(counts_after.values()) / max(1, min(counts_after.values()))
        print(f"\n  Imbalance ratio: {ratio_before:.1f}x -> {ratio_after:.1f}x")

    random.shuffle(balanced_rows)
    return balanced_rows


# ============================================================
# Step 2: Crop and save one image
# ============================================================
def process_one_image(item, image_root, output_dir, target_size):
    """Crop face, resize, and save. Returns True on success."""
    try:
        src_path = os.path.join(image_root, item["path"])
        src_path = os.path.normpath(src_path)

        if not os.path.exists(src_path):
            return False, "not_found"

        with Image.open(src_path) as img:
            # Convert to RGB if needed (some images are grayscale or RGBA)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Crop face region
            left = max(0, item["face_x"])
            top = max(0, item["face_y"])
            right = min(img.width, left + item["face_w"])
            bottom = min(img.height, top + item["face_h"])

            # Validate crop region
            if right <= left or bottom <= top:
                return False, "invalid_crop"

            cropped = img.crop((left, top, right, bottom))

            # Resize to target
            resized = cropped.resize(target_size, Image.LANCZOS)

            # Save
            class_name = EXPRESSION_MAP[item["expression"]]
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            filename = os.path.basename(item["path"])
            # Ensure .jpg extension
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += ".jpg"

            save_path = os.path.join(class_dir, filename)
            resized.save(save_path, "JPEG", quality=95)

            return True, "ok"

    except (IOError, UnidentifiedImageError, OSError) as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# ============================================================
# Step 3: Process all images with threading and progress
# ============================================================
def process_all(rows, image_root, output_dir, target_size, num_workers, label=""):
    """Process all rows with multi-threading and manual progress tracking."""
    print(f"\n  Processing {len(rows)} images ({label})...")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {num_workers}")

    os.makedirs(output_dir, exist_ok=True)

    success = 0
    failed = 0
    start_time = time.time()
    total = len(rows)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_one_image, item, image_root, output_dir, target_size): item
            for item in rows
        }

        for i, future in enumerate(as_completed(futures)):
            ok, msg = future.result()
            if ok:
                success += 1
            else:
                failed += 1

            # Print progress every 5000 images
            done = i + 1
            if done % 5000 == 0 or done == total:
                elapsed = time.time() - start_time
                speed = done / elapsed
                remaining = (total - done) / speed if speed > 0 else 0
                print(f"    [{done:>6}/{total}] "
                      f"success={success}, failed={failed}, "
                      f"speed={speed:.0f} img/s, "
                      f"ETA={remaining/60:.1f}min")

    elapsed = time.time() - start_time
    print(f"\n  Completed {label}: {success} saved, {failed} failed, "
          f"time={elapsed:.0f}s ({elapsed/60:.1f}min)")

    return success, failed


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  AffectNet Optimized Preprocessor")
    print("  Crop + Balance + Resize (224x224) + Multi-thread")
    print("=" * 60)

    # Process training set
    print("\n" + "=" * 60)
    print("  TRAINING SET")
    print("=" * 60)
    train_rows = load_and_balance_csv(TRAIN_CSV, MAX_PER_CLASS_TRAIN, "Training")
    train_output = os.path.join(OUTPUT_ROOT, "train")
    train_ok, train_fail = process_all(
        train_rows, IMAGE_ROOT, train_output, TARGET_SIZE, NUM_WORKERS, "Training"
    )

    # Process validation set
    print("\n" + "=" * 60)
    print("  VALIDATION SET")
    print("=" * 60)
    val_rows = load_and_balance_csv(VALID_CSV, MAX_PER_CLASS_VAL, "Validation")
    val_output = os.path.join(OUTPUT_ROOT, "val")
    val_ok, val_fail = process_all(
        val_rows, IMAGE_ROOT, val_output, TARGET_SIZE, NUM_WORKERS, "Validation"
    )

    # Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Training:   {train_ok:>6} images saved ({train_fail} failed)")
    print(f"  Validation: {val_ok:>6} images saved ({val_fail} failed)")
    print(f"  Total:      {train_ok + val_ok:>6} images")
    print(f"  Output:     {OUTPUT_ROOT}")
    print(f"\n  Structure:")
    print(f"    {OUTPUT_ROOT}/")
    print(f"      train/")
    for expr_id in range(8):
        print(f"        {EXPRESSION_MAP[expr_id]}/")
    print(f"      val/")
    for expr_id in range(8):
        print(f"        {EXPRESSION_MAP[expr_id]}/")
    print(f"\n  Ready for PyTorch ImageFolder!")
    print(f"  Usage: datasets.ImageFolder('{OUTPUT_ROOT}/train', transform=...)")
    print("=" * 60)


if __name__ == "__main__":
    main()
