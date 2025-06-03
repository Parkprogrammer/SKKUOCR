#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLOVA OCR 로 bbox·텍스트 추출 → crop 저장 & CSV 생성 (패딩 추가 버전)
------------------------------------------------------------------
INPUT  디렉토리
    train/merged_images  ,  test/merged_images
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
                           padding_ratio: float = 0.3) -> Tuple[int, int, int, int]:
    """
    bounding box를 패딩과 함께 확장
    
    Args:
        verts: OCR에서 반환된 4개의 꼭짓점 좌표
        img_height, img_width: 원본 이미지 크기
        padding_ratio: bbox 크기 대비 패딩 비율 (기본값: 30%)
    
    Returns:
        (x0, y0, x1, y1): 확장된 bounding box 좌표
    """
    # 좌표를 정수로 변환
    xs, ys = zip(*[(int(x), int(y)) for x, y in verts])
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    
    # 현재 bbox의 너비, 높이
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    
    # 패딩 계산 (bbox 크기에 비례)
    padding_x = int(bbox_width * padding_ratio)
    padding_y = int(bbox_height * padding_ratio)
    
    # 패딩 적용하여 확장 (모두 정수로 변환)
    x0_padded = int(max(0, x0 - padding_x))
    y0_padded = int(max(0, y0 - padding_y))
    x1_padded = int(min(img_width, x1 + padding_x))
    y1_padded = int(min(img_height, y1 + padding_y))
    
    return x0_padded, y0_padded, x1_padded, y1_padded


# ──────────────────────────────────────────────────────────────
# 1.  split(train / test) 1개 처리
# ──────────────────────────────────────────────────────────────
def process_split(split_name: str,
                  in_root: Path,
                  out_root: Path,
                  api_url: str, secret: str,
                  conf_th: float = 0.5,
                  padding_ratio: float = 0.3):
    """
    ex)
        process_split("train", Path("train"), Path("CLOVA_V2_train"), ...)
    """
    in_img_dir  = in_root / "merged_images"
    out_img_dir = out_root / "merged_images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Tuple[str,str]] = []
    out_index = 0

    img_paths = sorted(in_img_dir.glob("*"))
    print(f"[{split_name}] 처리할 이미지 수: {len(img_paths)}")
    
    for i, img_fp in enumerate(img_paths):
        if i % 10 == 0:
            print(f"[{split_name}] 진행률: {i}/{len(img_paths)}")
            
        resp = clova_ocr_api(img_fp, api_url, secret)
        crops = parse_fields(resp, conf_th)

        if not crops:                 # 아무 글자도 못 찾은 경우 pass
            print(f"[{split_name}] {img_fp.name}: OCR 결과 없음")
            continue

        # 원본 이미지를 BGR 로 읽는다 (crop 후 저장을 위해)
        img_bgr = cv2.imread(str(img_fp))
        if img_bgr is None:
            print(f"[WARN] cannot read {img_fp}")
            continue
            
        img_height, img_width = img_bgr.shape[:2]
        print(f"[{split_name}] {img_fp.name}: {len(crops)}개 텍스트 영역 검출")

        for text, conf, verts in crops:
            # 패딩과 함께 bbox 확장
            x0, y0, x1, y1 = expand_bbox_with_padding(
                verts, img_height, img_width, padding_ratio
            )
            
            crop = img_bgr[y0:y1, x0:x1]
            if crop.size == 0: 
                print(f"[WARN] 빈 crop 영역: {text}")
                continue
                
            # 최소 크기 체크 (너무 작은 crop 방지)
            crop_height, crop_width = crop.shape[:2]
            if crop_height < 10 or crop_width < 10:
                print(f"[WARN] 너무 작은 crop 무시: {text} ({crop_width}x{crop_height})")
                continue

            fname = f"{out_index:06d}.png"  # 파일명에 제로패딩 추가
            cv2.imwrite(str(out_img_dir/fname), crop)
            csv_rows.append((fname, text))
            out_index += 1

        # 너무 빠른 호출로 429 방지
        time.sleep(0.11)

    # CSV 저장 ---------------------------------------------------
    out_csv = out_root / f"{split_name}_labels.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["filename","text"])
        wr.writerows(csv_rows)
    print(f"[{split_name}] 완료: {len(csv_rows)}개 crop 저장 → {out_csv}")


# ──────────────────────────────────────────────────────────────
# 2.  main
# ──────────────────────────────────────────────────────────────
def main():
    load_dotenv()                         # .env 에  API_URL / SECRET_KEY 저장
    api_url   = os.environ["API_URL"]
    secret    = os.environ["SECRET_KEY"]

    # 패딩 설정 (bbox 크기 대비 비율)
    padding_ratio = 0.3  # 30% 패딩 (필요에 따라 조정 가능)
    
    print(f"패딩 비율: {padding_ratio*100}%")
    print("=" * 50)

    # (원본,  출력)  쌍 지정 - V2 디렉토리로 변경
    splits = [("train", Path("train"), Path("CLOVA_V2_train")),
              ("test" , Path("test") , Path("CLOVA_V2_test"))]

    for name, src, dst in splits:
        print(f"\n[시작] {name} 처리 중...")
        process_split(name, src, dst, api_url, secret, 
                     conf_th=0.5, padding_ratio=padding_ratio)
        print(f"[완료] {name} 처리 완료\n")


if __name__ == "__main__":
    main()