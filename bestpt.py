import argparse, json, cv2, torch, yaml, shutil, itertools
from pathlib import Path
from typing import List
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, Subset
from pororo.models.brainOCR.recognition import get_recognizer
from pororo.models.brainOCR import brainocr      # Reader 클래스
from pororo.tasks import download_or_load
from pororo.models.brainOCR.brainocr import Reader   # Reader 안에 이미 util 존재
from datasets import _BaseCrops, collate_eval, collate_train, evaluate_dataset, recognize_imgs1, recognize_imgs2
import torch.nn.functional as F
import re
import numpy as np
import os

def get_unique_save_dir(base_dir="assets", prefix="test"):
    os.makedirs(base_dir, exist_ok=True)  # assets 폴더가 없으면 생성

    idx = 1
    while True:
        save_dir = os.path.join(base_dir, f"{prefix}_{idx}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir
        idx += 1

FORBIDDEN = re.compile(r'[←→↔↕↖↗↘↙➔➜·●】≠○↑×■□▲△▼▽◇◆★]')  # 경고에 나온 특수 문자들을 추가
tbl = str.maketrans({"\n": " ", "\t": " "})  # 빠른 치환용 table
UNKNOWN_SET = set() 

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

        df = pd.read_csv(self.root / "merged_labels.csv", dtype={"text": str})
        for _, row in df.iterrows():
            img_path = self.root / "merged_images" / row["filename"]
            txt = row["text"]
            if not isinstance(txt, str) or not txt.strip():
                continue
            self.samples.append((img_path, txt))

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
        
        max_t = self.imgW // 4           # down-sampling 1/4 가정
        if length > max_t:
            return None 
        return img, text, length


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


def build_recognizer(opt_txt_fp: str, device: str = "cuda"):
    # 1) 옵션 txt 읽기
    opt = Reader.parse_options(opt_txt_fp)

    # 2) vocab 세팅
    opt["vocab"] = Reader.build_vocab(opt["character"])
    opt["vocab_size"] = len(opt["vocab"])
    opt["num_class"] = opt["vocab_size"]

    # 3) 모델 ckpt 경로 지정  (← 빠져 있어서 KeyError 발생)
    default_ckpt = "brainocr.pt"
    opt["rec_model_ckpt_fp"] = str(default_ckpt)

    # 4) 기타
    opt["device"] = device
    # opt["imgH"], opt["imgW"] = 64, 256

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

def recursive_train(model):
    model.train()
    for child in model.children():
        recursive_train(child)

def clean_text(text: str) -> str:
    # evaluate_dataset 내부에서 쓰이는 클린 함수와 동일 로직
    return re.sub(r"[^a-zA-Z0-9가-힣]", "", text).lower()

def evaluate_accuracy(recognizer, op2val, converter, test_loader, device="cuda"):
    """
    evaluate_dataset과 동일한 방식으로 배치별 GT∙PR을 비교하여 
    '공백 무시 + 소문자 + 한글·영어·숫자만 남김' 정확도를 계산합니다.
    """
    recognizer.eval()
    hit_cleaned = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    
    sum_counts = [0, 0, 0]  # 특수기호, 숫자, 기본 글자 카운트
    sum_conf_sums = [0.0, 0.0, 0.0]  # 각 카운트의 confidence 합계

    # 필요한 opt2val을 동적으로 생성 (imgH, imgW, batch_size, adjust_contrast)
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            # 배치 타입 확인
            if len(batch) == 3:  # train collate (imgs, tgt, tgt_len)
                imgs, tgt, tgt_len = batch
                imgs = imgs.to(device)

                # Loss 계산
                logits = recognizer(imgs)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                input_len = torch.full(
                    (imgs.size(0),), logits.size(1),
                    dtype=torch.long, device="cpu"
                )
                loss = criterion(log_probs.cpu(), tgt.cpu(), input_len, tgt_len.cpu())
                if not (torch.isinf(loss) or torch.isnan(loss)):
                    total_loss += loss.item()

                # 정확도 계산을 위해 GT 텍스트 디코딩
                if hasattr(converter, "decode"):
                    gt_texts = converter.decode(tgt, tgt_len)
                else:
                    gt_texts = converter.decode_greedy(tgt, tgt_len)

                # 예측 텍스트 생성
                img_list = (imgs[:, 0] * 255).byte().cpu().numpy()
                img_list = [img for img in img_list]
                preds, counts, sum_confs = recognize_imgs2(img_list, recognizer, converter, op2val)

                # 배치별 모델 통계 누적
                for i in range(3):
                    sum_counts[i] += counts[i]
                    sum_conf_sums[i] += sum_confs[i]

                for (pr_txt, conf), gt in zip(preds, gt_texts):
                    gt_clean = clean_text(gt)
                    pr_clean = clean_text(pr_txt)
                    if gt_clean == pr_clean:
                        hit_cleaned += 1
                    total += 1

            else:  # eval collate (imgs, gt_texts)
                imgs, gt_texts = batch
                imgs = imgs.to(device)

                # 예측 수행
                img_list = (imgs[:, 0] * 255).byte().cpu().numpy()
                img_list = [img for img in img_list]
                preds, counts, sum_confs = recognize_imgs2(img_list, recognizer, converter, op2val)

                # 배치별 모델 통계 누적
                for i in range(3):
                    sum_counts[i] += counts[i]
                    sum_conf_sums[i] += sum_confs[i]

                # 정확도 계산
                for (pr_txt, conf), gt in zip(preds, gt_texts):
                    gt_clean = clean_text(gt)
                    pr_clean = clean_text(pr_txt)
                    if gt_clean == pr_clean:
                        hit_cleaned += 1
                    total += 1

    # 전체 평균 confidence 계산
    overall_avg_confs = [
        (sum_conf_sums[i] / sum_counts[i]) if sum_counts[i] else 0.0
        for i in range(3)
    ]

    if total == 0:
        return 0.0, 0.0

    print(f"전체 테스트 데이터 모델별 (count/avg_conf) 특수기호: {sum_counts[0]}/{overall_avg_confs[0]:.3f}, 숫자: {sum_counts[1]}/{overall_avg_confs[1]:.3f}, 기본: {sum_counts[2]}/{overall_avg_confs[2]:.3f}")

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    return hit_cleaned / total, avg_loss, sum_counts, overall_avg_confs

# --------------------------------------------------------------------------
# 3. 파인튜닝 함수
# --------------------------------------------------------------------------
def finetune(recognizer, op2val, converter, train_loader, valid_loader, test_loader, epochs, lr, save_dir: Path, device="cuda"):
    # 손실∙옵티마이저∙스케줄러 세팅
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    optimizer = torch.optim.Adam(
        recognizer.parameters(), lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    recognizer.to(device)
    recognizer.train()

    best_acc = 0.0  # 최고 validation 정확도 기록

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        batch_count = 0

        # Training phase
        recognizer.train()
        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            imgs, tgt, tgt_len = batch
            imgs = imgs.to(device)

            recognizer.zero_grad()
            try:
                logits = recognizer(imgs)  # (B, T, C)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                input_len = torch.full(
                    (imgs.size(0),), logits.size(1),
                    dtype=torch.long, device="cpu"
                )
                loss = criterion(log_probs.cpu(), tgt.cpu(), input_len, tgt_len.cpu())

                if torch.isinf(loss) or torch.isnan(loss):
                    print(f"[skip] ep{ep} step{step}  loss={loss.item()}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(recognizer.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1
            except RuntimeError as e:
                print(f"[error skip] {e}")
                continue

        # epoch별 평균 training loss
        avg_train_loss = running_loss / batch_count if batch_count > 0 else float("nan")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[epoch {ep}/{epochs}] train_loss={avg_train_loss:.6f}, lr={current_lr:.2e}")

        # ── Validation phase ─────────────────────────────────────────────────
        val_acc, val_loss, val_sum_counts, val_avg_confs = evaluate_accuracy(recognizer, op2val, converter, valid_loader, device=device)
        print(f"--> [Validation] epoch {ep}: accuracy={val_acc*100:.2f}%, loss={val_loss:.6f}")
        
        test_acc, test_loss, test_sum_counts, test_avg_confs = evaluate_accuracy(recognizer, op2val, converter, test_loader, device=device)
        print(f"--> [Test] epoch {ep}: accuracy={test_acc*100:.2f}%, loss={test_loss:.6f}")
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # WandB 로깅 (nested dictionary structure for grouping)
        wandb.log({
            "epoch": ep,
            "train": {
                "loss": avg_train_loss,
                "accuracy": val_acc,  # training accuracy는 validation과 동일하게 처리
            },
            "validation": {
                "accuracy": val_acc,
                "loss": val_loss,
                "count_model3": val_sum_counts[0],
                "count_model2": val_sum_counts[1],
                "count_model1": val_sum_counts[2],
                "avg_conf_model3": val_avg_confs[0],
                "avg_conf_model2": val_avg_confs[1],
                "avg_conf_model1": val_avg_confs[2],
            },
            "test": {
                "accuracy": test_acc,
                "loss": test_loss,
                "count_model3": test_sum_counts[0] if 'test_sum_counts' in locals() else None,
                "count_model2": test_sum_counts[1] if 'test_sum_counts' in locals() else None,
                "count_model1": test_sum_counts[2] if 'test_sum_counts' in locals() else None,
                "avg_conf_model3": test_avg_confs[0] if 'test_avg_confs' in locals() else None,
                "avg_conf_model2": test_avg_confs[1] if 'test_avg_confs' in locals() else None,
                "avg_conf_model1": test_avg_confs[2] if 'test_avg_confs' in locals() else None,
            },
            "lr": current_lr
        })

        # ── 최고 validation 정확도 갱신 시 best.pt 저장 ──────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            best_fp = save_dir / "best.pt"
            torch.save(recognizer.state_dict(), best_fp)
            print(f"🟢 New best model saved (epoch {ep}, val_acc={val_acc*100:.2f}%) → {best_fp}")

        # 다시 학습 모드로 변환
        recursive_train(recognizer)

    print(f"★ Training 끝. Best validation accuracy={best_acc*100:.2f}%")
    return



# --------------------------------------------------------------------------
# 4. 저장 & 재로드 테스트
# --------------------------------------------------------------------------
def save_ckpt(recognizer, opt_dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_fp = save_dir / "finetune_clova.pt"
    torch.save(recognizer.state_dict(), ckpt_fp)

    opt_fp = save_dir / "finetune_clova_opt.txt"
    with opt_fp.open("w", encoding="utf-8") as f:
        for k, v in opt_dict.items():
            if k == "device":         # runtime 정보는 제외
                continue
            f.write(f"{k}: {v}\n")
    print(f"✓ Saved ckpt → {ckpt_fp}\n✓ Saved opt  → {opt_fp}")
    return ckpt_fp, opt_fp


def quick_eval(reader, data_loader, device="cuda", n_show=3):
    reader.recognizer.eval()

    blank, idx2char = 0, reader.converter.idx2char
    has_decode = hasattr(reader.converter, "decode")

    for batch in data_loader:
        if batch is None:
            continue

        # ── 2-tuple ? 3-tuple ? ─────────────────────────────
        if len(batch) == 3:                  # (imgs, tgt_idx, tgt_len)
            imgs, tgt, tgt_len = batch
            str_gt_available = False
        else:                                # (imgs, gt_text_list)
            imgs, gt_texts = batch
            str_gt_available = True

        imgs = imgs.to(device)
        with torch.no_grad():
            logits = reader.recognizer(imgs)

        preds_idx  = logits.softmax(2).argmax(2)
        preds_size = torch.full(
            (preds_idx.size(0),), preds_idx.size(1),
            dtype=torch.long, device=preds_idx.device)

        # ── idx → text ──────────────────────────────────────
        if has_decode:
            pred_txts, _ = reader.converter.decode(preds_idx, preds_size)
        else:                               # manual greedy
            pred_txts = []
            for seq, T in zip(preds_idx, preds_size):
                prev, chs = blank, []
                for t in range(T):
                    idx = seq[t].item()
                    if idx != blank and idx != prev:
                        chs.append(idx2char[idx])
                    prev = idx
                pred_txts.append("".join(chs))

        # ── GT 텍스트 얻기 ──────────────────────────────────
        if str_gt_available:
            gts = gt_texts
        else:
            if has_decode:
                gts, _ = reader.converter.decode(tgt, tgt_len)
            else:
                gts = reader.converter.decode_greedy(tgt, tgt_len)

        # ── print some ─────────────────────────────────────
        for gt, pr in zip(gts[:n_show], pred_txts[:n_show]):
            print(f"GT: {gt}\nPR: {pr}\n")
        break

def split_train_valid(dataset, val_ratio=0.2, random_state=42):
    """
    데이터셋을 train/validation으로 분리
    """
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_ratio, random_state=random_state
    )
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"데이터 분리 완료: Train={len(train_subset)}, Valid={len(val_subset)}")
    return train_subset, val_subset

def train_and_evaluate(epochs, batch_size, lr, args):
    save_dir = Path(get_unique_save_dir(base_dir=args.save_dir, prefix="finetune"))
    print(f"모델 저장 경로: {save_dir}")

    wandb.init(
        project="brainocr-fine-tuning",
        name=f"{save_dir.name}_ep{epochs}_lr{lr}_bs{batch_size}",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "device": args.device,
            "opt_txt": args.opt_txt,
            "train_root": args.train_root,
            "test_root": args.test_root,
            "val_ratio": args.val_ratio
        }
    )

    # ① recognizer/ converter 생성
    rec, converter, opt_dict = build_recognizer(args.opt_txt, device=args.device)

    # ② 전체 train 셋 준비
    full_train_set = _BaseCrops(
        csv_fp=Path(args.train_root) / "train_labels.csv",
        img_dir=Path(args.train_root) / "merged_images",
        img_size=(256, 64),
        converter=converter,
        for_train=True,
    )
    full_train_set.preload_for_stats()
    full_train_set.print_filter_report()

    # ③ train/validation 분리
    train_subset, val_subset = split_train_valid(full_train_set, val_ratio=args.val_ratio)

    # ④ test 셋 준비
    test_set = _BaseCrops(
        csv_fp=Path(args.test_root) / "test_labels.csv",
        img_dir=Path(args.test_root) / "merged_images",
        img_size=(256, 64),
        for_train=False,
    )
    test_set.preload_for_stats()
    test_set.print_filter_report()

    # ⑤ DataLoader 생성
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_train,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_train,  # validation도 loss 계산을 위해 train collate 사용
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_eval,
        pin_memory=True
    )

    # ⑥ finetuning 시작 (validation 사용)
    finetune(
        recognizer=rec,
        op2val=opt_dict,
        converter=converter,
        train_loader=train_loader,
        valid_loader=val_loader,  # validation loader 사용
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        save_dir=Path(save_dir),
        device=args.device
    )
    
    save_ckpt(rec, opt_dict, save_dir)

    # ⑦ 훈련 완료 후 최종 테스트 (best 모델 로드)
    print("\n" + "="*50)
    print("최종 테스트 시작...")
    print("="*50)
    
    final_reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp="/workspace/SKKUOCR2/SKKUOCR_fine/assets/craft.pt",
        rec_model_ckpt_fp=str(Path(save_dir) / "best.pt"),
        opt_fp=str(Path(save_dir) / "finetune_clova_opt.txt"),
        device=args.device,
    )
    final_reader.recognizer.to(args.device)

    # 최종 테스트 정확도 계산
    test_acc, test_loss, test_sum_counts, test_avg_confs  = evaluate_accuracy(final_reader.recognizer, opt_dict, converter, test_loader, device=args.device)
    print(f"🎯 최종 테스트 결과: accuracy={test_acc*100:.2f}%")
    
    # WandB에 최종 테스트 결과 로깅
    wandb.log({
        "final_test_accuracy": test_acc,
        "final_test_loss": test_loss
    })

    # 테스트 결과 상세 저장
    evaluate_dataset(final_reader, test_loader, device=args.device,
                     save_csv=str(Path(save_dir) / "test_pred.csv"))

    wandb.finish()
    print(f"✅ 모든 과정 완료! 결과는 {save_dir}에 저장됨")


def test_only(model_ckpt_fp, opt_fp, test_root, device="cuda"):
    # Reader 초기화
    reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp="/workspace/SKKUOCR2/SKKUOCR_fine/assets/craft.pt",
        rec_model_ckpt_fp=str(model_ckpt_fp),
        opt_fp=str(opt_fp),
        device=device,
    )
    reader.recognizer.to(device)

    # 테스트셋 로드
    test_set = _BaseCrops(
        csv_fp=Path(test_root) / "test_labels.csv",
        img_dir=Path(test_root) / "merged_images",
        img_size=(256, 64),
        for_train=False,
    )
    test_set.preload_for_stats()
    test_set.print_filter_report()

    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=False,
        num_workers=2, collate_fn=collate_eval, pin_memory=True
    )

    # 평가 수행 및 결과 저장
    evaluate_dataset(reader, test_loader, device=device, save_csv="assets/test_pred.csv")

    print("✅ 테스트 완료. 결과는 assets/test_pred.csv에 저장됨.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_txt",   default="ocr-opt.txt")
    parser.add_argument("--train_root", default="CLOVA_V3_train")
    parser.add_argument("--test_root",  default="CLOVA_V2_test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="assets")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    args = parser.parse_args()
    
    train_and_evaluate(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        args=args
    )

    # 만약 오로지 테스트만 돌리고 싶다면 아래를 사용하십시오.
    # test_only(
    #     model_ckpt_fp="assets/test_20/finetune_clova.pt",
    #     opt_fp="assets/test_20/finetune_clova_opt.txt",
    #     test_root="CLOVA_V2_test",
    #     device="cuda"
    # )