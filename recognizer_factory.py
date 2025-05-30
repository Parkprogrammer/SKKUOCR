#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1) pororo.models.brainOCR.recognition.get_recognizer() 로 기존 ckpt 로드
2) Handwriting Crops(Dataset) + CTC 학습
3) finetuned ckpt 와 opt(txt) 저장
4) 저장물을 Reader 로 다시 불러 테스트

run:  python finetune_brainocr.py --train_root correction_data_a/train/handwriting \
                                  --test_root  correction_data_a/test/handwriting \
                                  --epochs 5 --batch 32 --save_dir assets
"""
import argparse, json, cv2, torch, yaml, shutil
from pathlib import Path
from typing import List

from torch.utils.data import Dataset, DataLoader
from pororo.models.brainOCR.recognition import get_recognizer
from pororo.models.brainOCR import brainocr      # Reader 클래스
from pororo.tasks import download_or_load
from pororo.models.brainOCR.brainocr import Reader   # Reader 안에 이미 util 존재

import re

FORBIDDEN = r'[←→↔↕↖↗↘↙➔➜]'      # 필요에 따라 추가
tbl = str.maketrans({"\n": " ", "\t": " "})  # 빠른 치환용 table
UNKNOWN_SET = set()          # 학습 시작 전에 한 번 비워둡니다

# --------------------------------------------------------------------------
# 1. 데이터셋 + collate
# --------------------------------------------------------------------------
class HandwritingCrops(Dataset):
    """
    root/
      ├─ label/xxxx_metadata.json
      ├─ *.png ...
    """

    def __init__(self,
                 root_dir: str,
                 converter,
                 use_corrected: bool = True,
                 img_size: tuple = (100, 64)) -> None:
        self.root = Path(root_dir)
        self.imgW, self.imgH = img_size
        self.converter = converter
        self.samples = []  # [(png_path, text)]

        for meta_fp in (self.root / "label").glob("*.json"):
            metas = json.loads(meta_fp.read_text(encoding="utf-8"))
            for m in metas:
                txt = m["corrected_text"] if use_corrected else m["original_text"]
                if not txt.strip():
                    continue
                self.samples.append((self.root / m["image_filename"], txt))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        png_fp, label = self.samples[idx]

        # ---------- 이미지 ----------
        img = cv2.imread(str(png_fp), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.imgW, self.imgH), interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.

        # ---------- 텍스트 전처리 ----------
        cleaned = (label.translate(tbl)            # \n, \t → space
                         .strip())
        cleaned = re.sub(FORBIDDEN, " ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)

        # ---------- 허용 글자 체크 ----------
        if any(ch not in self.converter.char2idx for ch in cleaned):
            # ※ 전역으로 모아 두고 싶다면 여기에  UNKNOWN_SET.update(...)
            return None     # ← DataLoader에서 버려질 샘플

        text, length = self.converter.encode([cleaned])
        return img, text, length


def collate_ctc(batch):
    batch = [b for b in batch if b is not None]     # ← 필터
    if not batch:                                   # 전부 skip 된 경우
        return None

    imgs      = torch.stack([b[0] for b in batch])
    targets   = torch.cat([b[1] for b in batch])
    tgt_lens  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, targets, tgt_lens


# --------------------------------------------------------------------------
# 2. recognizer 로더 (opt.txt → dict → get_recognizer)
# --------------------------------------------------------------------------
def load_opt(opt_txt: str) -> dict:
    opt2val = {}
    for ln in Path(opt_txt).read_text(encoding="utf-8").splitlines():
        if ": " not in ln:
            continue
        k, v = ln.split(": ", 1)
        try:
            opt2val[k] = yaml.safe_load(v)         # 숫자·bool 도 형 변환
        except Exception:
            opt2val[k] = v
    return opt2val


# def build_recognizer(opt_txt_fp: str, device: str = "cuda"):
#     # ① txt 파싱 → dict
#     opt = Reader.parse_options(opt_txt_fp)          # {"character": "...", ...}

#     # ② vocab, vocab_size 추가
#     opt["vocab"] = Reader.build_vocab(opt["character"])
#     opt["vocab_size"] = len(opt["vocab"])

#     # ③ 기타 파라미터 덮어쓰기(필요 시)
#     opt["device"] = device       # gpu / cpu
#     opt["num_gpu"] = torch.cuda.device_count()     # 멀티 GPU라면
#     opt["num_class"] = opt["vocab_size"]

#     # ④ recognizer & converter 얻기
#     model, converter = get_recognizer(opt)          # <- 이제 KeyError 안 남
#     return model.to(device), converter, opt
def build_recognizer(opt_txt_fp: str, device: str = "cuda"):
    # 1) 옵션 txt 읽기
    opt = Reader.parse_options(opt_txt_fp)

    # 2) vocab 세팅
    opt["vocab"] = Reader.build_vocab(opt["character"])
    opt["vocab_size"] = len(opt["vocab"])
    opt["num_class"] = opt["vocab_size"]

    # 3) 모델 ckpt 경로 지정  (← 빠져 있어서 KeyError 발생)
    default_ckpt = Path.home() / ".pororo" / "misc" / "brainocr.pt"
    opt["rec_model_ckpt_fp"] = str(default_ckpt)

    # 4) 기타
    opt["device"] = device

    # 5) recognizer, converter 반환
    model, converter = get_recognizer(opt)
    model.to(device)
    
    # ---------- [NEW] : decode() 가 없으면 decode_greedy 로 래핑 ----------
    if not hasattr(converter, "decode"):
        import types
        def _decode(self, flat_idx, len_tensor):
            return self.decode_greedy(flat_idx, len_tensor)
        converter.decode = types.MethodType(_decode, converter)
    # ----------------------------------------------------------------------
    return model, converter, opt


# --------------------------------------------------------------------------
# 3. 파인튜닝 함수
# --------------------------------------------------------------------------
def finetune(recognizer, converter, train_loader,
             epochs: int = 5, device: str = "cuda"):
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    optimizer = torch.optim.Adam(recognizer.parameters(), lr=1e-4)

    for ep in range(1, epochs + 1):
        for imgs, tgt, tgt_len in train_loader:
            imgs = imgs.to(device) # tgt, tgt_len = imgs.to(device), tgt.to(device), tgt_len.to(device)
            logits = recognizer(imgs)                   # (B, T, vocab)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_len = torch.full(
                (imgs.size(0),), logits.size(1), dtype=torch.long, device=device
            )

            loss = criterion(log_probs, tgt, input_len, tgt_len)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recognizer.parameters(), 5.)
            optimizer.step()

        print(f"[epoch {ep}/{epochs}] loss={loss.item():.4f}")


# --------------------------------------------------------------------------
# 4. 저장 & 재로드 테스트
# --------------------------------------------------------------------------
def save_ckpt(recognizer, opt_dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_fp = save_dir / "finetuned_brainocr.pt"
    torch.save(recognizer.state_dict(), ckpt_fp)

    opt_fp = save_dir / "finetuned_opt.txt"
    with opt_fp.open("w", encoding="utf-8") as f:
        for k, v in opt_dict.items():
            if k == "device":         # runtime 정보는 제외
                continue
            f.write(f"{k}: {v}\n")
    print(f"✓ Saved ckpt → {ckpt_fp}\n✓ Saved opt  → {opt_fp}")
    return ckpt_fp, opt_fp


def quick_eval(reader, test_loader, device="cuda", n_show=3):
    reader.recognizer.eval()

    for imgs, tgt, tgt_len in test_loader:
        if imgs is None:          # collate_ctc 가 None 반환한 경우
            continue

        imgs = imgs.to(device)

        with torch.no_grad():
            logits = reader.recognizer(imgs)          # (B, T, V)

        # ---------------- best-path CTC --------------------------
        preds_idx  = logits.softmax(2).argmax(2)      # (B, T)
        preds_size = torch.full(
            size=(preds_idx.size(0),),
            fill_value=preds_idx.size(1),
            dtype=torch.long,
            device=preds_idx.device,
        )

        # ---------------- idx → text ----------------------------
        if hasattr(reader.converter, "decode"):
            pred_txts, _ = reader.converter.decode(preds_idx, preds_size)

        else:  # fallback – 수동 greedy CTC decode
            blank = 0
            idx2char = reader.converter.idx2char      # list: idx → char
            pred_txts = []
            for seq, T in zip(preds_idx, preds_size):
                prev = blank
                chars = []
                for i in range(T):
                    idx = seq[i].item()
                    if idx != blank and idx != prev:
                        chars.append(idx2char[idx])
                    prev = idx
                pred_txts.append("".join(chars))

        # ---------------- GT ------------------------------------
        if hasattr(reader.converter, "decode"):
            gts, _ = reader.converter.decode(tgt, tgt_len)   # 기존 converter.decode()
        else:                                                # fallback
            gts = reader.converter.decode_greedy(tgt, tgt_len)

        for gt, pr in zip(gts[:n_show], pred_txts[:n_show]):
            print(f"GT: {gt}\nPR: {pr}\n")
        break




# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_txt",   default="ocr-opt.txt")
    parser.add_argument("--train_root", required=True)
    parser.add_argument("--test_root",  required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="assets")
    args = parser.parse_args()

    # recognizer (pre-trained) ------------------------------------------------
    rec, converter, opt_dict = build_recognizer(args.opt_txt, device=args.device)

    # dataloader -------------------------------------------------------------
    train_ds = HandwritingCrops(args.train_root, converter)
    test_ds  = HandwritingCrops(args.test_root,  converter)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=4, collate_fn=collate_ctc, drop_last=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=2, collate_fn=collate_ctc, pin_memory=True
    )

    # fine-tune --------------------------------------------------------------
    finetune(rec, converter, train_loader, epochs=args.epochs, device=args.device)

    # save  ------------------------------------------------------------------
    ckpt_fp, opt_fp = save_ckpt(rec, opt_dict, Path(args.save_dir))

    # reload with Reader -----------------------------------------------------
    reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp=ckpt_fp.parent / "craft.pt",   # CRAFT 그대로
        rec_model_ckpt_fp=str(ckpt_fp),
        opt_fp=str(opt_fp),
        device=args.device,
    )
    reader.recognizer.to(args.device)
    quick_eval(reader, test_loader, args.device)
