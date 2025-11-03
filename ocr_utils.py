import os
import json
import base64
import time
import uuid
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''

    Here, we saved many util functions so that the Higher API model could 
    make labels for cropped images.
    
    Pipeline:
        1. <Base efficient model> -> <no label>
        2. Bring CLOVA_OCR API as a sort of Teacher model (The best for Korean character recognition)
        3. Create labels with the Teacher model

            <Base model> -> crop -> cropped images -> Recognizer    -> <Noisy Label> :
                                                    ↳ Teacher Model -> <Clean label> : Teacher label
            
        4. Use clean labels to create label for training Base models!

'''


def call_ocr(img_path: Path, api_url: str, secret: str) -> Optional[Dict]:
    """Call CLOVA OCR API for single image"""
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    req = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "lang": "ko",
        "images": [{
            "format": img_path.suffix[1:],
            "name": img_path.stem,
            "data": img_b64
        }],
        "enableTableDetection": False
    }
    headers = { "X-OCR-SECRET": secret, "Content-Type": "application/json" }
    res = requests.post(api_url, headers=headers, data=json.dumps(req))
    if res.status_code == 200:
        return res.json()
    print(f"[{img_path.name}] OCR failed: {res.status_code}")
    return None


def parse_ocr(resp: Dict, conf_th: float = 0.5) -> List[Tuple[str, float, List[Tuple[int, int]]]]:
    """
    Parse OCR response
    Returns: [(text, confidence, bbox_vertices[4]), ...]
    """
    if not resp or resp["images"][0]["inferResult"] != "SUCCESS":
        return []
    fields = resp["images"][0].get("fields", [])
    out = []
    for f in fields:
        c = f["inferConfidence"]
        if c < conf_th:
            continue
        verts = [(v["x"], v["y"]) for v in f["boundingPoly"]["vertices"]]
        out.append((f["inferText"], c, verts))
    return out

# This function is for padding the bbox to ensure the CLOVA understadns the image.
def pad_bbox(verts: List[Tuple[int, int]], img_h: int, img_w: int,
             pad_ratio: float = 0.3, min_pad: int = 10) -> Tuple[int, int, int, int]:
    """Expand bbox with padding"""
    xs, ys = zip(*[(int(x), int(y)) for x, y in verts])
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    
    pad_x = max(min_pad, int(bbox_w * pad_ratio))
    pad_y = max(min_pad, int(bbox_h * pad_ratio))
    
    x0_pad = max(0, x0 - pad_x)
    y0_pad = max(0, y0 - pad_y)
    x1_pad = min(img_w, x1 + pad_x)
    y1_pad = min(img_h, y1 + pad_y)
    
    return x0_pad, y0_pad, x1_pad, y1_pad

# This function apply augmentation so that, 
# the CLOVA model understands the image with higher confidence.
def enhance_crop(crop: np.ndarray, 
                 target_h: int = 64,
                 target_w: Optional[int] = None,
                 contrast: bool = True,
                 sharpen: bool = True,
                 denoise: bool = True) -> np.ndarray:
    """Enhance crop image quality"""
    if crop.size == 0:
        return crop
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    if contrast:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    if sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
    
    h, w = gray.shape
    if target_w is None:
        aspect = w / h
        target_w = int(target_h * aspect)
    
    if target_h > h or target_w > w:
        enhanced = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    else:
        enhanced = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return enhanced


def save_crop(crop: np.ndarray, 
              save_path: str,
              level: str = "medium") -> bool:
    """Save enhanced crop"""
    if crop.size == 0:
        return False
    
    settings = {
        "light": {"target_h": 64, "contrast": True, "sharpen": False, "denoise": False},
        "medium": {"target_h": 96, "contrast": True, "sharpen": True, "denoise": True},
        "heavy": {"target_h": 128, "contrast": True, "sharpen": True, "denoise": True}
    }
    
    config = settings.get(level, settings["medium"])
    enhanced = enhance_crop(crop, **config)
    success = cv2.imwrite(save_path, enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    return success


def valid_crop(text: str, crop_w: int, crop_h: int, 
               img_w: int, img_h: int) -> Tuple[bool, str]:
    """Check if crop meets quality standards"""
    text_clean = text.strip()
    if len(text_clean) == 0:
        return False, f"Empty text: '{text}'"
    
    if len(text_clean) == 1:
        if text_clean in '→-ㆍ.,;(){}[]<>\\~`\'\"_=+!?@#$%^&*|/':
            return False, f"Meaningless char: '{text}'"
    
    if len(text_clean) > 20:
        return False, f"Too long: '{text[:20]}...' (len: {len(text_clean)})"
    
    base_w, base_h = 1920, 1080
    w_scale = img_w / base_w
    h_scale = img_h / base_h
    
    min_h = max(5, int(12 * h_scale))
    min_w = max(5, int(15 * w_scale))
    if crop_h < min_h or crop_w < min_w:
        return False, f"Too small: {text} ({crop_w}x{crop_h}, min: {min_w}x{min_h})"
    
    aspect = crop_w / crop_h
    if aspect > 20:
        return False, f"Too wide: {text} (ratio: {aspect:.2f})"
    
    if aspect < 0.05:
        return False, f"Too narrow: {text} (ratio: {aspect:.2f})"
    
    return True, "PASS"


def viz_bbox(img_path: Path, crops: List[Tuple[str, float, List[Tuple[int, int]]]], 
             img_bgr: np.ndarray, pad_ratio: float = 0.3) -> None:
    """Visualize OCR results and bboxes"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_bgr.shape[:2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(img_rgb)
    ax1.set_title(f'Original Bboxes: {img_path.name}\n({len(crops)} detections)', fontsize=14)
    ax1.axis('off')
    
    ax2.imshow(img_rgb)
    ax2.set_title(f'Padded Bboxes (padding: {pad_ratio*100}%)', fontsize=14)
    ax2.axis('off')
    
    valid_cnt = 0
    filtered_cnt = 0
    
    for i, (text, conf, verts) in enumerate(crops):
        xs, ys = zip(*[(int(x), int(y)) for x, y in verts])
        x0_orig, y0_orig, x1_orig, y1_orig = min(xs), min(ys), max(xs), max(ys)
        
        rect1 = patches.Rectangle((x0_orig, y0_orig), x1_orig-x0_orig, y1_orig-y0_orig, 
                                   linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect1)
        ax1.text(x0_orig, y0_orig-5, f'{i}: {text[:10]}', fontsize=8, color='red', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        x0_pad, y0_pad, x1_pad, y1_pad = pad_bbox(verts, img_h, img_w, pad_ratio)
        
        crop_w = x1_pad - x0_pad
        crop_h = y1_pad - y0_pad
        is_valid, reason = valid_crop(text, crop_w, crop_h, img_w, img_h)
        
        if is_valid:
            color = 'green'
            valid_cnt += 1
            status = '✓'
        else:
            color = 'orange'
            filtered_cnt += 1
            status = '✗'
        
        rect2 = patches.Rectangle((x0_pad, y0_pad), crop_w, crop_h, 
                                   linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect2)
        ax2.text(x0_pad, y0_pad-5, f'{i}{status}: {text[:10]}', fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    fig.suptitle(f'Valid: {valid_cnt}, Filtered: {filtered_cnt}, Total: {len(crops)}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n[DEBUG] {img_path.name} detailed analysis:")
    for i, (text, conf, verts) in enumerate(crops):
        x0_pad, y0_pad, x1_pad, y1_pad = pad_bbox(verts, img_h, img_w, pad_ratio)
        crop_w = x1_pad - x0_pad
        crop_h = y1_pad - y0_pad
        is_valid, reason = valid_crop(text, crop_w, crop_h, img_w, img_h)
        
        status = "✓ PASS" if is_valid else f"✗ FILTER: {reason}"
        print(f"  {i:2d}. '{text}' ({crop_w}x{crop_h}, conf:{conf:.2f}) -> {status}")
    print()


def get_images(base_path: Path) -> List[Path]:
    """Collect all images from category folders"""
    categories = ['image', 'notice', 'handwriting']
    all_imgs = []
    
    for cat in categories:
        cat_path = base_path / cat
        if cat_path.exists():
            exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            for ext in exts:
                all_imgs.extend(cat_path.glob(ext))
            print(f"  {cat}: {len(list(cat_path.glob('*')))} files found")
        else:
            print(f"  {cat}: folder not exists")
    
    return sorted(all_imgs)