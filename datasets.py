# datasets.py
import re, cv2, torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import csv
from pororo import Pororo
from difflib import SequenceMatcher


from typing import Optional
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import DataLoader
from typing import List, Sequence, Tuple
from pororo.models.brainOCR.recognition import AlignCollate, ListDataset,               \
                         recognizer_predict, second_recognizer_predict



FORBIDDEN = re.compile(r'[←→↔↕↖↗↘↙➔➜]')   # ← ① compile!
_tbl = str.maketrans({"\n": " ", "\t": " "})

class _BaseCrops(Dataset):
    """CSV 한 장을 받아 <img,tgt> 를 뱉는 공통 로직"""
    def __init__(self, *, csv_fp: Path, img_dir: Path,
                 img_size=(100, 64), converter=None, for_train=True):

        # ---------- 1) CSV 읽기 (헤더 유무 자동 감지) ----------
        try:
            head = pd.read_csv(csv_fp, nrows=1)
            has_header = 'filename' in head.columns.str.lower()
        except Exception as e:
            raise RuntimeError(f"CSV 읽기 실패: {csv_fp}\n{e}")

        if has_header:
            df = pd.read_csv(
                csv_fp,
                dtype={'filename': str, 'text': str, 'category': str},
                keep_default_na=False
            )
            df.columns = ['filename', 'text', 'category']   # 컬럼 표준화
        else:
            df = pd.read_csv(
                csv_fp, header=None,
                names=['filename', 'text', 'category'],
                dtype={'filename': str, 'text': str, 'category': str},
                keep_default_na=False
            )

        # ---------- 2) 헤더행·빈 텍스트 제거 ----------
        df = df[df['filename'].str.lower() != 'filename']
        df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)

        # ---------- 3) 실제 파일 존재 여부 필터 ----------
        exists_mask = df['filename'].apply(lambda fn: (img_dir / fn).is_file())
        missing = len(df) - exists_mask.sum()
        if missing:
            print(f"⚠️  {missing} labels removed (file not found)")
        df = df[exists_mask].reset_index(drop=True)

        # ---------- 4) 정보 출력 & 멤버 저장 ----------
        print(f"Loaded {len(df)} samples from {csv_fp}")
        print(f"Sample data:\n{df.head()}")

        self.df        = df
        self.img_dir   = img_dir
        self.imgW, self.imgH = img_size
        self.converter = converter
        self.for_train = for_train

    def __len__(self): return len(self.df)

    def _clean(self, txt: str):
        txt = txt.translate(_tbl)
        txt = FORBIDDEN.sub(" ", txt)
        txt = re.sub(r"\s{2,}", " ", txt).strip()
        return txt

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_fp  = self.img_dir / row['filename']
        gt_text = self._clean(row['text'])

        # --- 이미지 로드 & resize ---------------------------------
        img = cv2.imread(str(img_fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Cannot load image {img_fp}")                         
            return None
        
        img = cv2.resize(img, (self.imgW, self.imgH),
                         interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.

        if not self.for_train:                       # 평가용 → (img, str)
            return img, gt_text

        # 학습용 → (img, encoded, len)
        if any(ch not in self.converter.char2idx for ch in gt_text):
            print(f"Warning: Unknown characters in text '{gt_text}'")
            return None                              # DataLoader 에서 skip

        enc, ln = self.converter.encode([gt_text])
        # CTC length check  (down-sample ratio ≈ 4)
        if ln > self.imgW // 4:
            print(f"Warning: Text too long for image width. Text length: {ln}, Max allowed: {self.imgW // 4}")
            return None
        return img, enc, ln

def collate_train(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, tgt, tgt_len = zip(*batch)
    return (torch.stack(imgs),
            torch.cat(tgt),
            torch.tensor(tgt_len, dtype=torch.long))

def collate_eval(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels = zip(*batch)
    return torch.stack(imgs), list(labels)


def recognize_imgs(img_list      : Sequence[np.ndarray],
                   recognizer,
                   converter,
                   opt2val : dict,
                   conf_th : float = 0.2
                  ) -> List[Tuple[str,float]]:
    """
    detector 없이, 이미지 crop 들(회색·RGB 아무거나)만 받아 글자를 읽어 낸다.
    반환 값: [(텍스트, confidence), ...]  ―  입력 img_list 와 같은 순서
    """
    imgH, imgW          = opt2val["imgH"],  opt2val["imgW"]
    # adjust_contrast     = opt2val["adjust_contrast"]
    adjust_contrast     = 0.5
    batch_size          = opt2val["batch_size"]
    # n_workers           = opt2val["n_workers"]
    n_workers           = 0

    # ① 1st pass ──────────────────────────────────────
    collate   = AlignCollate(imgH, imgW, adjust_contrast)
    dl        = DataLoader(ListDataset(img_list), batch_size,
                           False, num_workers=n_workers,
                           collate_fn=collate, pin_memory=True)
    result1   = second_recognizer_predict(recognizer, converter, dl, opt2val)

    # ② 2nd pass (저신뢰만) ───────────────────────────
    low_idx   = [i for i,(txt,conf) in enumerate(result1) if conf < conf_th]
    if low_idx:
        img2 = [img_list[i] for i in low_idx]
        dl2  = DataLoader(ListDataset(img2), batch_size,
                          False, num_workers=n_workers,
                          collate_fn=collate, pin_memory=True)
        result2 = recognizer_predict(recognizer, converter, dl2, opt2val)

        # 결과 머지
        for k,i in enumerate(low_idx):
            # 더 높은 confidence 만 취한다
            if result2[k][1] > result1[i][1]:
                result1[i] = result2[k]

    # result1 : [(text, conf), ...]
    return result1

def evaluate_dataset(reader,
                     data_loader,
                     device: str = "cuda",
                     save_csv: Optional[str] = None):
    """
    전체 data_loader 를 돌며
      GT, PR 을 모두 출력하고 최종 accuracy 를 계산한다.

    Args
    ----
    reader     : brainocr.Reader (recognizer+converter 포함)
    data_loader: collate_eval 을 쓰는 DataLoader
                 (batch 가  (imgs, str-list) 형태여야 합니다)
    save_csv   : "결과를 csv 로 저장할 경로"; 생략하면 콘솔만 출력
    """
    recog, conv = reader.recognizer, reader.converter
    opt         = reader.opt2val               # imgH·imgW 등 들어 있음
    recog.eval()

    hit = tot = 0
    fp   = open(save_csv, "w", newline="", encoding="utf-8") if save_csv else None
    wr   = csv.writer(fp) if fp else None
    if wr:
        wr.writerow(["ground_truth", "prediction", "confidence"])

    # ────────────────────────────────────────────────────────────────
    for imgs, gts in data_loader:          # imgs : (B,1,H,W) / gts list[str]
        B = imgs.size(0)

        # ★ 1) Tensor → numpy 이미지 리스트 --------------------------
        img_list: List[np.ndarray] = (imgs[:, 0] * 255).byte().cpu().numpy()
        img_list = [img for img in img_list]           # (H,W) uint8 배열들

        # ★ 2) OCR --------------------------------------------------
        preds = recognize_imgs(img_list, recog, conv, opt)   # [(txt, conf), …]

        # ★ 3) 집계·출력 --------------------------------------------
        for (pr_txt, conf), gt in zip(preds, gts):
            ok  = int(gt.replace(" ", "") == pr_txt.replace(" ", ""))
            hit += ok;  tot += 1

            print(f"GT: {gt}\nPR: {pr_txt}\nCONF:{conf:.3f}  "
                  f"{'✓' if ok else '✗'}\n{'-'*40}")

            if wr:
                wr.writerow([gt, pr_txt, f"{conf:.4f}"])

    # ────────────────────────────────────────────────────────────────
    acc = hit / tot if tot else 0
    print(f"\n✅  accuracy = {hit}/{tot}  ({acc:.2%})")
    if fp:
        fp.close()
        print(f"🔖  CSV saved to:  {save_csv}")
        
        
def normalize_text(text: str) -> str:
    """Normalize text for comparison - 기존 checker.py와 동일한 방식"""
    # Remove extra whitespace, newlines, special characters
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
    text = re.sub(r'[^\w\s가-힣]', '', text)  # Remove special chars except Korean
    return text.strip().lower()

def compare_texts(text1: str, text2: str) -> Tuple[bool, bool, float]:
    """Compare two texts with multiple criteria - 기존 checker.py와 동일한 방식"""
    # 1. Exact match
    exact_match = text1.strip() == text2.strip()
    
    # 2. Normalized match (ignore spacing, special chars)
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    similar_match = norm1 == norm2
    
    # 3. Similarity score using SequenceMatcher
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    return exact_match, similar_match, similarity

def evaluate_ocr_vs_gpt_simple(csv_path: str) -> dict:
    """
    CSV에서 GPT 결과와 새로운 OCR 결과를 비교
    
    Args:
        csv_path: OCR vs GPT 결과 CSV 파일 경로
    
    Returns:
        dict: 정확도 통계
    """
    
    # OCR 초기화
    print("Loading OCR model...")
    ocr = Pororo(task="ocr", lang="ko", model="brainocr")
    
    # CSV 읽기
    df = pd.read_csv(csv_path, keep_default_na=False)  # NaN 방지
    
    # 이미지별로 그룹화
    image_groups = df.groupby('image_path')
    
    total_exact = 0
    total_similar = 0
    total_similarity = 0.0
    total_count = 0
    
    for image_path, group in image_groups:
        print(f"Processing: {image_path}")
        
        # 새로운 OCR 실행
        try:
            ocr_result = ocr(image_path, detail=True)
            if not ocr_result.get('bounding_poly'):
                continue
        except Exception as e:
            print(f"OCR failed: {e}")
            continue
        
        # 각 bbox 비교
        for _, row in group.iterrows():
            bbox_index = int(row['bbox_index'])
            gpt_text = str(row['gpt_text'])
            
            # NaN 값 처리
            if gpt_text in ['nan', 'NaN', ''] or pd.isna(row['gpt_text']):
                print(f"  Bbox {bbox_index}: Skipping - GPT text is empty or NaN")
                continue
            
            # OCR 결과에서 해당 bbox 텍스트 가져오기
            if bbox_index < len(ocr_result['bounding_poly']):
                new_ocr_text = ocr_result['bounding_poly'][bbox_index]['description']
                
                # 비교 - 기존 checker.py와 동일한 방식 사용
                exact, similar, similarity = compare_texts(new_ocr_text, gpt_text)
                
                total_exact += int(exact)
                total_similar += int(similar)
                total_similarity += similarity
                total_count += 1
                
                # print(f"  Bbox {bbox_index}: OCR='{new_ocr_text[:20]}...' | GPT='{gpt_text[:20]}...' | "
                #       f"Exact={exact} | Similar={similar} | Score={similarity:.3f}")
    
    # 결과 계산
    if total_count == 0:
        return {"error": "No valid comparisons found"}
    
    exact_accuracy = (total_exact / total_count) * 100
    similar_accuracy = (total_similar / total_count) * 100
    avg_similarity = total_similarity / total_count
    
    results = {
        "total_comparisons": total_count,
        "exact_matches": total_exact,
        "similar_matches": total_similar,
        "exact_accuracy": exact_accuracy,
        "similar_accuracy": similar_accuracy, 
        "average_similarity": avg_similarity,
        "exact_accuracy_percent": exact_accuracy,
        "similar_accuracy_percent": similar_accuracy,
        "high_similarity_matches": sum(1 for i in range(total_count) if (total_similarity / total_count) >= 0.8),  # 임시값
    }
    
    print(f"\n{'='*50}")
    print("OCR vs GPT Comparison Results (checker.py와 동일한 방식)")
    print(f"{'='*50}")
    print(f"Total bounding boxes: {total_count:,}")
    print(f"\nAccuracy Metrics:")
    print(f"  Exact matches: {total_exact:,} ({exact_accuracy:.2f}%)")
    print(f"  Similar matches: {total_similar:,} ({similar_accuracy:.2f}%)")
    print(f"  Average similarity: {avg_similarity:.3f}")
    print(f"{'='*50}")
    
    return results

# 사용 예시
if __name__ == "__main__":
    # 사용법
    results = evaluate_ocr_vs_gpt_simple("test_2_ocr_gpt_results_org.csv")
    
    # 여러 파일 처리
    csv_files = ["train_2_ocr_gpt_results_org.csv", "test_2_ocr_gpt_results_org.csv"]
    
    # for csv_file in csv_files:
    #     print(f"\n=== {csv_file} ===")
    #     try:
    #         evaluate_ocr_vs_gpt_simple(csv_file)
    #     except FileNotFoundError:
    #         print(f"File not found: {csv_file}")
    #     except Exception as e:
    #         print(f"Error: {e}")