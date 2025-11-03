import os
import csv
import time
from pathlib import Path
from typing import List, Tuple

import cv2
from dotenv import load_dotenv

from pororo.models.brainOCR import brainocr
from datasets import recognize_imgs
from ocr_utils import (
    call_ocr, parse_ocr, pad_bbox, save_crop, 
    valid_crop, viz_bbox, get_images
)

'''
    Creating datset [Cropped images with teacher label for finetuning the efficient model]

'''


def debug_mode(split_name: str, in_root: Path,
               api_url: str, secret: str,
               conf_th: float = 0.5,
               pad_ratio: float = 0.3,
               debug_cnt: int = 5):
    """Debug mode for bbox visualization"""
    img_paths = get_images(in_root)[:debug_cnt]
    print(f"[DEBUG] {split_name} -> {len(img_paths)} images test")

    for i, img_fp in enumerate(img_paths):
        print(f"\n[DEBUG {i+1}/{len(img_paths)}] Processing {img_fp.name} ({img_fp.parent.name})...")

        img_bgr = cv2.imread(str(img_fp))
        if img_bgr is None:
            print("  Image load failed")
            continue

        resp = call_ocr(img_fp, api_url, secret)
        crops = parse_ocr(resp, conf_th)

        if not crops:
            print("  No OCR results")
            continue

        viz_bbox(img_fp, crops, img_bgr, pad_ratio)

        key = input("Enter=continue   q=quit   s=run all : ").strip().lower()
        if key == "q":
            return False
        elif key == "s":
            return True
        time.sleep(0.2)
    
    return True


def process_split(split_name: str,
                  in_root: Path,
                  out_root: Path,
                  api_url: str, 
                  secret: str,
                  conf_th: float = 0.5,
                  pad_ratio: float = 0.3,
                  enhance_level: str = "medium",
                  reader=None,
                  min_rec_conf: float = 0.18):
    """Process split with enhanced crop handling"""
    if reader:
        recog = reader.recognizer
        converter = reader.converter
        opt2val = reader.opt2val
        
    out_img_dir = out_root / "merged_images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Tuple[str, str, str]] = []
    out_idx = 0
    
    stats = {
        'total_detected': 0,
        'filtered_out': 0,
        'saved': 0,
        'enhancement_failed': 0,
        'ocr_failed': 0,
        'image_load_failed': 0,
        'filter_reasons': {},
        'category_stats': {'image': 0, 'notice': 0, 'handwriting': 0}
    }

    img_paths = get_images(in_root)
    print(f"[{split_name}] Total images to process: {len(img_paths)}")
    print(f"[{split_name}] Enhancement level: {enhance_level}")
    
    out_csv = out_root / f"{split_name}_labels_2.csv"
    
    for i, img_fp in enumerate(img_paths):
        print(f"\n[{split_name}] Progress: {i+1}/{len(img_paths)} - {img_fp.name} ({img_fp.parent.name})")
        
        cat = img_fp.parent.name
        
        try:
            print(f"  1) Running OCR...")
            resp = call_ocr(img_fp, api_url, secret)
            
            if resp is None:
                print(f"  -> OCR failed, skip")
                stats['ocr_failed'] += 1
                continue
                
            crops = parse_ocr(resp, conf_th)
            print(f"  -> {len(crops)} text regions detected")

            if not crops:
                print(f"  -> No text detected, skip")
                continue

            print(f"  2) Loading image...")
            img_bgr = cv2.imread(str(img_fp))
            if img_bgr is None:
                print(f"  -> Image load failed, skip")
                stats['image_load_failed'] += 1
                continue
                
            img_h, img_w = img_bgr.shape[:2]
            print(f"  -> Image size: {img_w}x{img_h}")

            print(f"  3) Processing crops...")
            valid_crops = 0
            
            for j, (text, conf, verts) in enumerate(crops):
                stats['total_detected'] += 1
                
                print(f"    Processing: '{text}' (conf: {conf:.2f})")
                
                x0, y0, x1, y1 = pad_bbox(verts, img_h, img_w, pad_ratio)
                
                crop = img_bgr[y0:y1, x0:x1]
                if crop.size == 0: 
                    print(f"    -> Empty crop, skip")
                    stats['filtered_out'] += 1
                    continue
                    
                crop_h, crop_w = crop.shape[:2]
                is_valid, reason = valid_crop(text, crop_w, crop_h, img_w, img_h)
                
                if not is_valid:
                    print(f"    -> Filtered: {reason}")
                    stats['filtered_out'] += 1
                    filter_type = reason.split(':')[0]
                    stats['filter_reasons'][filter_type] = stats['filter_reasons'].get(filter_type, 0) + 1
                    continue
                
                if reader:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                    pr_txt, rec_conf = recognize_imgs([gray], recog, converter, opt2val)[0]
                    if rec_conf < min_rec_conf:
                        print(f"    -> recognizer conf {rec_conf:.2f} < {min_rec_conf}, skip")
                        stats['filtered_out'] += 1
                        stats['filter_reasons']['low_rec_conf'] = \
                            stats['filter_reasons'].get('low_rec_conf', 0) + 1
                        continue

                fname = f"{out_idx:06d}.png"
                save_path = str(out_img_dir / fname)
                
                if save_crop(crop, save_path, enhance_level):
                    csv_rows.append((fname, text, cat))
                    out_idx += 1
                    stats['saved'] += 1
                    stats['category_stats'][cat] += 1
                    valid_crops += 1
                else:
                    stats['enhancement_failed'] += 1

            print(f"  -> {valid_crops} crops saved")
            
        except KeyboardInterrupt:
            print(f"\nInterrupted by user. Saving current results...")
            if csv_rows:
                with out_csv.open("w", newline="", encoding="utf-8") as fp:
                    wr = csv.writer(fp)
                    wr.writerow(["filename", "text", "category"])
                    wr.writerows(csv_rows)
                print(f"Saved until interruption: {out_csv}")
            break
        except Exception as e:
            print(f"  Error occurred: {e}")
            print(f"  -> Continue to next image...")
            continue

        print(f"  4) Waiting for API rate limit...")
        time.sleep(0.5)
        
        if (i + 1) % 500 == 0:
            print(f"  Updating CSV... ({i + 1}/{len(img_paths)})")
            with out_csv.open("w", newline="", encoding="utf-8") as fp:
                wr = csv.writer(fp)
                wr.writerow(["filename", "text", "category"])
                wr.writerows(csv_rows)
            print(f"  CSV updated: {out_csv} ({len(csv_rows)} entries)")

    print(f"\n[{split_name}] Final CSV saving...")
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["filename", "text", "category"])
        wr.writerows(csv_rows)
    
    print(f"\n[{split_name}] Processing complete statistics:")
    print(f"  Total detected: {stats['total_detected']}")
    print(f"  Filtered out: {stats['filtered_out']}") 
    print(f"  Saved crops: {stats['saved']}")
    print(f"  Enhancement failed: {stats['enhancement_failed']}")
    print(f"  Enhancement level: {enhance_level}")
    
    print(f"  Category-wise saved:")
    for cat, cnt in stats['category_stats'].items():
        print(f"    {cat}: {cnt}")
    
    if stats['filter_reasons']:
        print(f"  Filter reasons:")
        for reason, cnt in stats['filter_reasons'].items():
            print(f"    {reason}: {cnt}")
    
    print(f"  CSV saved: {out_csv}")
    print(f"  Images saved: {out_img_dir}")


def main():
    load_dotenv()
    api_url = os.environ["API_URL"]
    secret = os.environ["SECRET_KEY"]
    
    reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp="pororo/misc/craft.pt",
        rec_model_ckpt_fp="assets/finetune_clova.pt",
        opt_fp="assets/finetune_clova_opt.txt",
        device="cuda",
    )
    
    min_rec_conf = float(input("brainOCR min confidence? (default 0.18): ") or 0.18)

    print("=== Crop Image Enhancement Settings ===")
    print("1. light   - 64px height, basic contrast")
    print("2. medium  - 96px height, contrast+sharpen+denoise")  
    print("3. heavy   - 128px height, all enhancements")
    
    enhance_choice = input("Select enhancement level (1-3, default 2): ").strip()
    enhance_map = {"1": "light", "2": "medium", "3": "heavy"}
    enhance_level = enhance_map.get(enhance_choice, "medium")
    
    pad_input = input("Padding ratio (0.01-0.5, default 0.3): ").strip()
    try:
        pad_ratio = float(pad_input)
        pad_ratio = max(0.01, min(0.5, pad_ratio))
    except:
        pad_ratio = 0.01
    
    print(f"\nSelected settings:")
    print(f"  Enhancement level: {enhance_level}")
    print(f"  Padding ratio: {pad_ratio*100}%")
    
    enhance_details = {
        "light": "64px, contrast only",
        "medium": "96px, contrast+sharpen+denoise", 
        "heavy": "128px, all enhancements"
    }
    print(f"  Details: {enhance_details[enhance_level]}")
    print("=" * 50)
    
    debug = input("\nRun debug mode for bbox check? (y/N): ").strip().lower() == 'y'
    
    splits = [
        ("train", Path("train_2"), Path("CLOVA_V2_train")),
        ("test", Path("test_2"), Path("CLOVA_V2_test"))
    ]

    for name, src, dst in splits:
        print(f"\n[Start] Processing {name}...")
        print(f"Input path: {src}")
        
        total_imgs = get_images(src)
        if not total_imgs:
            print(f"No images found in {src}!")
            continue
            
        print(f"Total {len(total_imgs)} images found")
        
        if debug:
            should_continue = debug_mode(name, src, api_url, secret, 
                                        conf_th=0.5, pad_ratio=pad_ratio)
            if not should_continue:
                continue
        
        process_split(name, src, dst, api_url, secret,
                     conf_th=0.5,
                     pad_ratio=pad_ratio,
                     enhance_level=enhance_level,
                     reader=reader,
                     min_rec_conf=min_rec_conf)
        print(f"[Complete] {name} processing done\n")

    print("\nAll processing complete!")
    print(f"Enhanced crop images saved (level: {enhance_level})")


if __name__ == "__main__":
    main()