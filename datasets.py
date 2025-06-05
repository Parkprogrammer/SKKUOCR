# datasets.py
import re, cv2, torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import csv

from typing import Optional
import torch.nn.functional as F
import numpy as np 
from torch.utils.data import DataLoader
from typing import List, Sequence, Tuple
from pororo.models.brainOCR.recognition import AlignCollate, ListDataset,               \
                         recognizer_predict, second_recognizer_predict
import wandb


FORBIDDEN = re.compile(r'[←→↔↕↖↗↘↙➔➜·●】≠○↑×■□▲△▼▽◇◆★]')  # 경고에 나온 특수 문자들을 추가
_tbl = str.maketrans({"\n": " ", "\t": " "})  # 빠른 치환용 table
class _BaseCrops(Dataset):
    def __init__(self, *, csv_fp: Path, img_dir: Path,
                 img_size=(100, 64), converter=None, for_train=True):

        df = pd.read_csv(csv_fp, header=None, names=["filename", "text"],
                         dtype={"filename": str, "text": str},
                         keep_default_na=False)
        df = df[df["filename"].str.lower() != "filename"]
        df = df[df["text"].fillna("").str.strip().astype(bool)].reset_index(drop=True)

        self.df = df
        self.img_dir = img_dir
        self.imgW, self.imgH = img_size
        self.converter = converter
        self.for_train = for_train

        # 통계 카운터 추가
        self.stats = {
            "total": len(df),
            "img_load_fail": 0,
            "img_shape_filter": 0,
            "text_len_filter": 0,
            "char_invalid": 0,
            "ctc_len_filter": 0,
            "passed": 0
        }

    def __len__(self):
        return len(self.df)

    def _clean(self, txt: str):
        txt = txt.translate(_tbl)
        txt = FORBIDDEN.sub(" ", txt)
        txt = re.sub(r"\s{2,}", " ", txt).strip()
        return txt

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_fp = self.img_dir / row['filename']
        gt_text = self._clean(row['text'])

        img = cv2.imread(str(img_fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.stats["img_load_fail"] += 1
            return None

        h0, w0 = img.shape
        if w0 * h0 < 300 :
            self.stats["img_shape_filter"] += 1
            return None

        img = cv2.resize(img, (self.imgW, self.imgH), interpolation=cv2.INTER_AREA)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.

        if len(gt_text) <= 1:
            self.stats["text_len_filter"] += 1
            return None

        if not self.for_train:
            self.stats["passed"] += 1
            return img, gt_text

        if any(ch not in self.converter.char2idx for ch in gt_text):
            self.stats["char_invalid"] += 1
            return None

        enc, ln = self.converter.encode([gt_text])
        if ln > self.imgW // 4:
            self.stats["ctc_len_filter"] += 1
            return None

        self.stats["passed"] += 1
        return img, enc, ln

    def print_filter_report(self):
        print("=== Data Filtering Report ===")
        total = self.stats["total"]
        for k, v in self.stats.items():
            if k == "total": continue
            print(f"{k:>18}: {v:>5} ({v/total*100:.2f}%)")
        print("==============================")
        
    def preload_for_stats(self):
        for i in range(len(self)):
            _ = self.__getitem__(i)

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

def clean_text(text: str) -> str:
    # 영어, 숫자, 한글만 남기고 제거
    return re.sub(r"[^a-zA-Z0-9가-힣]", "", text).lower()

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

    hit = tot = hit_cleaned = 0
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
            ok  = int(gt.replace(" ", "").lower() == pr_txt.replace(" ", "").lower()) #공백 무시하고 평가
            hit += ok;  tot += 1
            
            gt_cleaned = clean_text(gt); pr_cleaned = clean_text(pr_txt)
            ok_cleaned = int(gt_cleaned.replace(" ", "") == pr_cleaned.replace(" ", ""))
            hit_cleaned += ok_cleaned

            print(f"GT: {gt}\nPR: {pr_txt}\nCONF:{conf:.3f}  "
                  f"{'✓' if ok else '✗'}\n{'-'*40}")

            if wr:
                wr.writerow([gt, pr_txt, f"{conf:.4f}"])

    # ────────────────────────────────────────────────────────────────
    acc = hit / tot if tot else 0
    acc_cleaned = hit_cleaned / tot if tot else 0
    
    print(f"\n✅  accuracy = {hit}/{tot}  ({acc:.2%})")
    print(f"🧹  특수기호 제거 accuracy = {hit_cleaned}/{tot}  ({acc_cleaned:.2%})")
    if fp:
        fp.close()
        print(f"🔖  CSV saved to:  {save_csv}")
        wandb.log({"eval/accuracy": acc, "eval/accuracy_cleaned": acc_cleaned})
        