import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

test_img_dir = Path("train_clova/merged_images")
img_exts = [".png", ".jpg", ".jpeg", ".bmp"]
widths, heights, areas = [], [], []

for img_fp in test_img_dir.iterdir():
    if img_fp.suffix.lower() in img_exts:
        img = cv2.imread(str(img_fp))
        if img is None:
            continue  
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        areas.append(w * h)

def save_hist(data, title, xlabel, filename):
    plt.figure()
    plt.hist(data, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    quantiles = np.percentile(data, [5, 25, 50, 75, 95])
    print(f"{title} Quantiles:")
    print(f"  5%  : {quantiles[0]:.1f}")
    print(f"  25% : {quantiles[1]:.1f}")
    print(f"  50% : {quantiles[2]:.1f}")
    print(f"  75% : {quantiles[3]:.1f}")
    print(f"  95% : {quantiles[4]:.1f}")
    print()

# 히스토그램 저장
save_hist(widths, "Width Distribution", "Width (pixels)", "hist_width.png")
save_hist(heights, "Height Distribution", "Height (pixels)", "hist_height.png")
save_hist(areas, "Area Distribution", "Area (pixels²)", "hist_area.png")

print("save image in hist_width.png, hist_height.png, hist_area.png")