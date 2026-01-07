

import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor ,SamAutomaticMaskGenerator
import time

import sys
import os
import urllib.request

def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "-" * (bar_len - filled)

        sys.stdout.write(
            f"\r[{bar}] {percent:6.2f}% "
            f"({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)"
        )
        sys.stdout.flush()


def download_sam_vit_b(checkpoint_path="sam_vit_b_01ec64.pth"):
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / 1e6
        if size_mb < 100:   # way too small to be real
            print("Checkpoint corrupted. Re-downloading...")
            os.remove(checkpoint_path)
        else:
            print("SAM ViT-B checkpoint already exists.")
            return

    print("Downloading SAM ViT-B checkpoint...")
    urllib.request.urlretrieve(url, checkpoint_path, reporthook=progress)
    print("\nDownload complete.")




# -------- Load image --------
def object_segment(path):
    download_sam_vit_b()
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -------- Generate masks --------
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    H, W, _ = image.shape
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9,
        min_mask_region_area=.1 * H * W,  
    )

    start_generate = time.time()
    masks = mask_generator.generate(image_rgb)
    elapsed_generate = time.time() - start_generate
    print(f"elapsed genertart : {elapsed_generate}")
    print(f"Total masks found: {len(masks)}")



    doc_id = 0
    doc_times = []

    start_total = time.time()

    for m in masks:
        start_m = time.time()

        x, y, w, h = map(int, m["bbox"])
        area = m["area"]

        #filtering
        if area < 0.2 * H * W:
            continue

        aspect_ratio = w / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        # apply mask
        mask = m["segmentation"]
        masked = image.copy()
        masked[~mask] = 255

        crop = masked[y:y+h, x:x+w]
        cv2.imwrite(f"result/document_{doc_id}.png", crop)

        doc_id += 1
        doc_times.append(time.time() - start_m)

    elapsed_total = time.time() - start_total
    # metrics
    avg_doc_time = sum(doc_times) / len(doc_times)
    docs_per_sec = doc_id / elapsed_total
    return doc_id,docs_per_sec,avg_doc_time,elapsed_total


doc_id,docs_per_sec,avg_doc_time,elapsed_total  = object_segment('input_images/single/image_20.jpg')
print(f"Exported documents: {doc_id}")
print(f"Avg time per document: {avg_doc_time:.4f} sec")
print(f"Documents per second: {docs_per_sec:.2f}")
print(f"Total elapsed time: {elapsed_total:.2f} sec")
