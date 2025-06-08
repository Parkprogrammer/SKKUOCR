#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLOVA OCR 로 bbox·텍스트 추출 → crop 저장 & CSV 생성 (멀티 카테고리 버전)
------------------------------------------------------------------
INPUT  디렉토리
    train_2/image, train_2/notice, train_2/handwriting
    test_2/image, test_2/notice, test_2/handwriting
OUTPUT 디렉토리
    CLOVA_V2_train/merged_images + CLOVA_V2_train/train_labels.csv
    CLOVA_V2_test/merged_images  + CLOVA_V2_test/test_labels.csv
"""

import os, json, base64, time, uuid, csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pororo.models.brainOCR import brainocr           # Reader
from pororo.models.brainOCR.recognition import AlignCollate, ListDataset
from datasets import recognize_imgs                   # 이미 작성돼 있음


# ──────────────────────────────────────────────────────────────
# 0.  CLOVA  OCR  호출 유틸
# ──────────────────────────────────────────────────────────────
def clova_ocr_api(img_path: Path, api_url: str, secret: str
                  ) -> Optional[Dict]:
    """단일 이미지 → CLOVA OCR 결과(JSON)"""
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    req = {
        "version":"V2","requestId":str(uuid.uuid4()),
        "timestamp":int(time.time()*1000),"lang":"ko",
        "images":[{"format":img_path.suffix[1:],
                   "name":img_path.stem, "data":img_b64}],
        "enableTableDetection": False
    }
    headers = {"X-OCR-SECRET": secret,
               "Content-Type":"application/json"}
    res = requests.post(api_url, headers=headers, data=json.dumps(req))
    if res.status_code==200: return res.json()
    print(f"[{img_path.name}] OCR 실패 → {res.status_code}")
    return None


def parse_fields(resp: Dict, conf_th: float = 0.5
                ) -> List[Tuple[str, float, List[Tuple[int, int]]]]:
    """
    반환: [(text, confidence, bbox_vertices[4]), ...]
    bbox 는 (x,y) int 튜플 4개
    """
    if not resp or resp["images"][0]["inferResult"]!="SUCCESS": return []
    fields = resp["images"][0].get("fields",[])
    out=[]
    for f in fields:
        c = f["inferConfidence"]
        if c < conf_th: continue
        verts = [(v["x"], v["y"]) for v in
                 f["boundingPoly"]["vertices"]]
        out.append((f["inferText"], c, verts))
    return out


def expand_bbox_with_padding(verts: List[Tuple[int, int]], 
                           img_height: int, img_width: int,
                           padding_ratio: float = 0.3,
                           min_padding: int = 10) -> Tuple[int, int, int, int]:
    """
    개선된 bbox 확장 함수 (최소 패딩 보장)
    """
    xs, ys = zip(*[(int(x), int(y)) for x, y in verts])
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    
    # 현재 bbox의 너비, 높이
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    
    # 패딩 계산 (비율 기반 + 최소값 보장)
    padding_x = max(min_padding, int(bbox_width * padding_ratio))
    padding_y = max(min_padding, int(bbox_height * padding_ratio))
    
    # 패딩 적용하여 확장
    x0_padded = max(0, x0 - padding_x)
    y0_padded = max(0, y0 - padding_y)
    x1_padded = min(img_width, x1 + padding_x)
    y1_padded = min(img_height, y1 + padding_y)
    
    return x0_padded, y0_padded, x1_padded, y1_padded


def enhance_crop_quality(crop: np.ndarray, 
                        target_height: int = 64,
                        target_width: Optional[int] = None,
                        enhance_contrast: bool = True,
                        sharpen: bool = True,
                        denoise: bool = True) -> np.ndarray:
    """
    crop 이미지의 품질을 향상시킵니다.
    """
    if crop.size == 0:
        return crop
    
    # 1) 그레이스케일 변환 (텍스트 인식에 더 좋음)
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    
    # 2) 노이즈 제거 (작은 잡음 제거)
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 3) 대비 향상 (텍스트를 더 선명하게)
    if enhance_contrast:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 추가 대비 조정
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    # 4) 샤프닝 (텍스트 가장자리를 더 선명하게)
    if sharpen:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
    
    # 5) 크기 조정 (더 큰 크기로 업스케일링)
    h, w = gray.shape
    if target_width is None:
        # 종횡비 유지하면서 높이 기준으로 조정
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
    
    # 업스케일링 시 INTER_CUBIC 사용 (더 부드러운 결과)
    if target_height > h or target_width > w:
        enhanced = cv2.resize(gray, (target_width, target_height), 
                            interpolation=cv2.INTER_CUBIC)
    else:
        enhanced = cv2.resize(gray, (target_width, target_height), 
                            interpolation=cv2.INTER_AREA)
    
    return enhanced


def save_enhanced_crop(crop: np.ndarray, 
                      save_path: str,
                      enhancement_level: str = "medium") -> bool:
    """
    향상된 crop을 저장합니다.
    """
    if crop.size == 0:
        return False
    
    # 향상 수준별 설정
    settings = {
        "light": {
            "target_height": 64,
            "enhance_contrast": True,
            "sharpen": False,
            "denoise": False
        },
        "medium": {
            "target_height": 96,  # 더 큰 크기
            "enhance_contrast": True,
            "sharpen": True,
            "denoise": True
        },
        "heavy": {
            "target_height": 128,  # 가장 큰 크기
            "enhance_contrast": True,
            "sharpen": True,
            "denoise": True
        }
    }
    
    config = settings.get(enhancement_level, settings["medium"])
    
    # 이미지 향상 처리
    enhanced = enhance_crop_quality(crop, **config)
    
    # PNG로 저장 (무손실 압축)
    success = cv2.imwrite(save_path, enhanced, 
                         [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 최고 품질
    
    return success


def is_valid_crop(text: str, crop_width: int, crop_height: int, img_width: int, img_height: int) -> Tuple[bool, str]:
    """
    crop이 품질 기준을 만족하는지 검사 (이미지 크기 비례 동적 조정)
    """
    # 텍스트 길이 체크
    text_clean = text.strip()
    if len(text_clean) == 0:
        return False, f"빈 텍스트: '{text}'"
    
    # 한 글자인 경우 특수문자만 제외 (숫자, 문자는 허용)
    if len(text_clean) == 1:
        if text_clean in '→-ㆍ.,;(){}[]<>\\~`\'\"_=+!?@#$%^&*|/':
            return False, f"의미없는 특수문자: '{text}'"
    
    if len(text_clean) > 20:
        return False, f"너무 긴 텍스트: '{text[:20]}...' (len: {len(text_clean)})"
    
    # 이미지 크기 기반 동적 임계값 계산
    # 기준 해상도: 1920x1080
    base_width, base_height = 1920, 1080
    width_scale = img_width / base_width
    height_scale = img_height / base_height
    
    # 최소 크기 (이미지 크기에 비례)
    min_height = max(5, int(12 * height_scale))
    min_width = max(5, int(15 * width_scale))
    if crop_height < min_height or crop_width < min_width:
        return False, f"너무 작은 crop: {text} ({crop_width}x{crop_height}, 기준: {min_width}x{min_height})"
    
    # 종횡비 체크 (일정)
    aspect_ratio = crop_width / crop_height
    if aspect_ratio > 20:
        return False, f"너무 가늘게 긴 crop: {text} (ratio: {aspect_ratio:.2f})"
    
    if aspect_ratio < 0.05:
        return False, f"너무 납작한 crop: {text} (ratio: {aspect_ratio:.2f})"
    
    return True, "PASS"


def visualize_bbox_detection(img_path: Path, crops: List[Tuple[str, float, List[Tuple[int, int]]]], 
                            img_bgr: np.ndarray, padding_ratio: float = 0.3) -> None:
    """
    OCR 결과와 bbox를 시각화하여 검증
    """
    # RGB로 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_bgr.shape[:2]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 원본 이미지 + 원래 bbox
    ax1.imshow(img_rgb)
    ax1.set_title(f'Original Bboxes: {img_path.name}\n({len(crops)} detections)', fontsize=14)
    ax1.axis('off')
    
    # 패딩 적용된 이미지 + 확장된 bbox
    ax2.imshow(img_rgb)
    ax2.set_title(f'Padded Bboxes (padding: {padding_ratio*100}%)', fontsize=14)
    ax2.axis('off')
    
    valid_count = 0
    filtered_count = 0
    
    for i, (text, conf, verts) in enumerate(crops):
        # 원래 bbox 그리기
        xs, ys = zip(*[(int(x), int(y)) for x, y in verts])
        x0_orig, y0_orig, x1_orig, y1_orig = min(xs), min(ys), max(xs), max(ys)
        
        rect1 = patches.Rectangle((x0_orig, y0_orig), x1_orig-x0_orig, y1_orig-y0_orig, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect1)
        ax1.text(x0_orig, y0_orig-5, f'{i}: {text[:10]}', fontsize=8, color='red', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # 패딩 적용된 bbox
        x0_pad, y0_pad, x1_pad, y1_pad = expand_bbox_with_padding(
            verts, img_height, img_width, padding_ratio
        )
        
        # 품질 검사
        crop_width = x1_pad - x0_pad
        crop_height = y1_pad - y0_pad
        is_valid, reason = is_valid_crop(text, crop_width, crop_height, img_width, img_height)
        
        if is_valid:
            color = 'green'
            valid_count += 1
            status = '✓'
        else:
            color = 'orange'
            filtered_count += 1
            status = '✗'
        
        rect2 = patches.Rectangle((x0_pad, y0_pad), crop_width, crop_height, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect2)
        ax2.text(x0_pad, y0_pad-5, f'{i}{status}: {text[:10]}', fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 통계 정보 추가
    fig.suptitle(f'Valid: {valid_count}, Filtered: {filtered_count}, Total: {len(crops)}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 필터링된 항목들의 이유 출력
    print(f"\n[DEBUG] {img_path.name} 상세 분석:")
    for i, (text, conf, verts) in enumerate(crops):
        x0_pad, y0_pad, x1_pad, y1_pad = expand_bbox_with_padding(
            verts, img_height, img_width, padding_ratio
        )
        crop_width = x1_pad - x0_pad
        crop_height = y1_pad - y0_pad
        is_valid, reason = is_valid_crop(text, crop_width, crop_height, img_width, img_height)
        
        status = "✓ PASS" if is_valid else f"✗ FILTER: {reason}"
        print(f"  {i:2d}. '{text}' ({crop_width}x{crop_height}, conf:{conf:.2f}) → {status}")
    print()


def get_all_images_from_categories(base_path: Path) -> List[Path]:
    """
    여러 카테고리 폴더에서 모든 이미지를 수집
    """
    categories = ['image', 'notice', 'handwriting']
    all_images = []
    
    for category in categories:
        category_path = base_path / category
        if category_path.exists():
            # 이미지 파일 확장자들
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            for ext in extensions:
                all_images.extend(category_path.glob(ext))
            print(f"  {category}: {len(list(category_path.glob('*')))}개 파일 발견")
        else:
            print(f"  {category}: 폴더가 존재하지 않음")
    
    return sorted(all_images)


def debug_mode_processing(split_name: str, in_root: Path,
                          api_url: str, secret: str,
                          conf_th: float = 0.5,
                          padding_ratio: float = 0.3,
                          debug_count: int = 5):
    """
    디버그 모드에서 bbox 시각화 및 검증
    """
    img_paths = get_all_images_from_categories(in_root)[:debug_count]
    print(f"[DEBUG] {split_name} → {len(img_paths)}장 테스트")

    for i, img_fp in enumerate(img_paths):
        print(f"\n[DEBUG {i+1}/{len(img_paths)}] {img_fp.name} ({img_fp.parent.name}) 처리 중...")

        # 1️⃣ 이미지 읽기 (원본 그대로)
        img_bgr = cv2.imread(str(img_fp))
        if img_bgr is None:
            print("  ⚠️ 이미지 읽기 실패")
            continue

        # 2️⃣ OCR 수행 (원본 이미지로)
        resp = clova_ocr_api(img_fp, api_url, secret)
        crops = parse_fields(resp, conf_th)

        if not crops:
            print("  OCR 결과 없음")
            continue

        # 3️⃣ bbox 시각화
        visualize_bbox_detection(img_fp, crops, img_bgr, padding_ratio)

        # 4️⃣ 계속 진행 여부 확인
        key = input("Enter=계속   q=종료   s=전체실행 : ").strip().lower()
        if key == "q":
            return False
        elif key == "s":
            return True
        time.sleep(0.2)
    
    return True


def process_split_enhanced(split_name: str,
                           in_root: Path,
                           out_root: Path,
                           api_url: str, secret: str,
                           conf_th: float = 0.5,
                           padding_ratio: float = 0.3,
                           enhancement_level: str = "medium",
                           reader=None,                # ★ 추가
                           min_rec_conf: float = 0.18  # ★ 추가
                           ):
   """
   향상된 crop 처리가 포함된 split 처리 함수 - 디버깅 강화
   """
   if reader:
     recog     = reader.recognizer
     converter = reader.converter
     opt2val   = reader.opt2val
        
   out_img_dir = out_root / "merged_images"
   out_img_dir.mkdir(parents=True, exist_ok=True)

   csv_rows: List[Tuple[str,str,str]] = []
   out_index = 0
   
   # 통계 카운터
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

   # 모든 카테고리에서 이미지 수집
   img_paths = get_all_images_from_categories(in_root)
   print(f"[{split_name}] 총 처리할 이미지 수: {len(img_paths)}")
   print(f"[{split_name}] 향상 수준: {enhancement_level}")
   
   # CSV 파일 경로 미리 설정
   out_csv = out_root / f"{split_name}_labels_2.csv"
   
   for i, img_fp in enumerate(img_paths):
       # 매 이미지마다 진행률 출력 (디버깅용)
       print(f"\n[{split_name}] 진행률: {i+1}/{len(img_paths)} - {img_fp.name} ({img_fp.parent.name})")
       
       category = img_fp.parent.name
       
       try:
           # OCR 수행
           print(f"  1) OCR 수행 중...")
           resp = clova_ocr_api(img_fp, api_url, secret)
           
           if resp is None:
               print(f"  → OCR 실패, 다음 이미지로...")
               stats['ocr_failed'] += 1
               continue
               
           crops = parse_fields(resp, conf_th)
           print(f"  → {len(crops)}개 텍스트 영역 검출")

           if not crops:
               print(f"  → 검출된 텍스트 없음, 다음 이미지로...")
               continue

           # 원본 이미지 읽기
           print(f"  2) 이미지 로드 중...")
           img_bgr = cv2.imread(str(img_fp))
           if img_bgr is None:
               print(f"  → 이미지 로드 실패, 다음 이미지로...")
               stats['image_load_failed'] += 1
               continue
               
           img_height, img_width = img_bgr.shape[:2]
           print(f"  → 이미지 크기: {img_width}x{img_height}")

           # crop 처리
           print(f"  3) crop 처리 중...")
           valid_crops = 0
           
           for j, (text, conf, verts) in enumerate(crops):
               stats['total_detected'] += 1
               
               print(f"    처리 중: '{text}' (conf: {conf:.2f})")
               
               # bbox 확장
               x0, y0, x1, y1 = expand_bbox_with_padding(
                   verts, img_height, img_width, padding_ratio
               )
               
               crop = img_bgr[y0:y1, x0:x1]
               if crop.size == 0: 
                   print(f"    → 빈 crop, 스킵")
                   stats['filtered_out'] += 1
                   continue
                   
               # 품질 검사
               crop_height, crop_width = crop.shape[:2]
               is_valid, reason = is_valid_crop(text, crop_width, crop_height, img_width, img_height)
               
               if not is_valid:
                   print(f"    → 필터링: {reason}")
                   stats['filtered_out'] += 1
                   filter_type = reason.split(':')[0]
                   stats['filter_reasons'][filter_type] = stats['filter_reasons'].get(filter_type, 0) + 1
                   continue
               
               if reader:
                    # crop → uint8 gray
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                    pr_txt, rec_conf = recognize_imgs([gray], recog, converter, opt2val)[0]
                    if rec_conf < min_rec_conf:
                        print(f"    → recognizer conf {rec_conf:.2f} < {min_rec_conf}, skip")
                        stats['filtered_out'] += 1
                        stats['filter_reasons']['low_rec_conf'] = \
                            stats['filter_reasons'].get('low_rec_conf', 0) + 1
                        continue

               # 향상된 crop 저장
               fname = f"{out_index:06d}.png"
               save_path = str(out_img_dir / fname)
               
               if save_enhanced_crop(crop, save_path, enhancement_level):
                   csv_rows.append((fname, text, category))
                   out_index += 1
                   stats['saved'] += 1
                   stats['category_stats'][category] += 1
                   valid_crops += 1
                   # print(f"    → 저장 성공: {fname}")
               else:
                   # print(f"    → 저장 실패: {text}")
                   stats['enhancement_failed'] += 1

           print(f"  → {valid_crops}개 crop 저장 완료")
           
       except KeyboardInterrupt:
           print(f"\n사용자에 의해 중단됨. 현재까지 결과 저장 중...")
           # 중단 시 현재까지 저장
           if csv_rows:
               with out_csv.open("w", newline="", encoding="utf-8") as fp:
                   wr = csv.writer(fp)
                   wr.writerow(["filename", "text", "category"])
                   wr.writerows(csv_rows)
               print(f"중단 시점까지 저장 완료: {out_csv}")
           break
       except Exception as e:
           print(f"  오류 발생: {e}")
           print(f"  → 다음 이미지로 계속...")
           continue

       # API 호출 간격 조정
       print(f"  4) API 호출 간격 대기...")
       time.sleep(0.5)
       
       # 500장마다 CSV 업데이트
       if (i + 1) % 500 == 0:
           print(f"  📝 CSV 업데이트 중... ({i + 1}/{len(img_paths)})")
           with out_csv.open("w", newline="", encoding="utf-8") as fp:
               wr = csv.writer(fp)
               wr.writerow(["filename", "text", "category"])
               wr.writerows(csv_rows)
           print(f"  ✅ CSV 업데이트 완료: {out_csv} ({len(csv_rows)}개 항목)")

   # 최종 CSV 저장
   print(f"\n[{split_name}] 최종 CSV 저장 중...")
   with out_csv.open("w", newline="", encoding="utf-8") as fp:
       wr = csv.writer(fp)
       wr.writerow(["filename", "text", "category"])
       wr.writerows(csv_rows)
   
   # 통계 출력
   print(f"\n[{split_name}] 처리 완료 통계:")
   print(f"  총 검출된 텍스트: {stats['total_detected']}개")
   print(f"  필터링된 텍스트: {stats['filtered_out']}개") 
   print(f"  저장된 crop: {stats['saved']}개")
   print(f"  향상 처리 실패: {stats['enhancement_failed']}개")
   print(f"  향상 수준: {enhancement_level}")
   
   print(f"  카테고리별 저장 통계:")
   for cat, count in stats['category_stats'].items():
       print(f"    {cat}: {count}개")
   
   if stats['filter_reasons']:
       print(f"  필터링 이유별 통계:")
       for reason, count in stats['filter_reasons'].items():
           print(f"    {reason}: {count}개")
   
   print(f"  CSV 저장: {out_csv}")
   print(f"  이미지 저장: {out_img_dir}")


# ──────────────────────────────────────────────────────────────
# 2.  main
# ──────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    api_url   = os.environ["API_URL"]
    secret    = os.environ["SECRET_KEY"]
    
    reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp="pororo/misc/craft.pt",          # detector 그대로
        rec_model_ckpt_fp="assets/finetune_clova.pt",      # ← 학습한 ckpt
        opt_fp="assets/finetune_clova_opt.txt",
        device="cuda",
    )
    
    min_rec_conf = float(input("brainOCR 최소 confidence? (기본 0.18): ") or 0.18)

    # 사용자 설정
    print("=== Crop 이미지 향상 설정 ===")
    print("1. light   - 64px 높이, 기본 대비 향상")
    print("2. medium  - 96px 높이, 대비+샤프닝+노이즈제거")  
    print("3. heavy   - 128px 높이, 모든 향상 기능")
    
    enhancement_choice = input("향상 수준 선택 (1-3, 기본값 2): ").strip()
    enhancement_map = {"1": "light", "2": "medium", "3": "heavy"}
    enhancement_level = enhancement_map.get(enhancement_choice, "medium")
    
    # 패딩 설정
    padding_input = input("패딩 비율 입력 (0.01-0.5, 기본값 0.3): ").strip()
    try:
        padding_ratio = float(padding_input)
        padding_ratio = max(0.01, min(0.5, padding_ratio))  # 0.1~0.5 범위로 제한
    except:
        padding_ratio = 0.01
    
    print(f"\n선택된 설정:")
    print(f"  향상 수준: {enhancement_level}")
    print(f"  패딩 비율: {padding_ratio*100}%")
    
    enhancement_details = {
        "light": "64px, 대비향상만",
        "medium": "96px, 대비+샤프닝+노이즈제거", 
        "heavy": "128px, 모든 향상기능"
    }
    print(f"  상세 설정: {enhancement_details[enhancement_level]}")
    print("=" * 50)
    
    # 디버그 모드 옵션
    debug_mode = input("\n디버그 모드로 bbox 확인? (y/N): ").strip().lower() == 'y'
    
    splits = [("train", Path("train_2"), Path("CLOVA_V2_train")),
              ("test" , Path("test_2") , Path("CLOVA_V2_test"))]

    for name, src, dst in splits:
        print(f"\n[시작] {name} 처리 중...")
        print(f"입력 경로: {src}")
        
        # 카테고리별 이미지 수 확인
        total_images = get_all_images_from_categories(src)
        if not total_images:
            print(f"⚠️  {src}에서 이미지를 찾을 수 없습니다!")
            continue
            
        print(f"총 {len(total_images)}개 이미지 발견")
        
        if debug_mode:
            # 디버그 모드로 시각화
            should_continue = debug_mode_processing(name, src, api_url, secret, 
                                                  conf_th=0.5, padding_ratio=padding_ratio)
            if not should_continue:
                continue
        
        # 향상된 처리 모드 사용
        process_split_enhanced(name, src, dst, api_url, secret,
                       conf_th=0.5,
                       padding_ratio=padding_ratio,
                       enhancement_level=enhancement_level,
                       reader=reader,              # ★ 추가
                       min_rec_conf=min_rec_conf)  # ★ 추가
        print(f"[완료] {name} 처리 완료\n")

    print("\n🎉 모든 처리 완료!")
    print(f"💡 향상된 crop 이미지들이 저장되었습니다 (수준: {enhancement_level})")


if __name__ == "__main__":
    main()