import argparse
import wandb
import torch
import pandas as pd
import cv2
import re
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pororo.models.brainOCR import brainocr
from pororo.models.brainOCR.brainocr import Reader

FORBIDDEN = r'[←→↔↕↖↗↘↙➔➜]'
tbl = str.maketrans({"\n": " ", "\t": " "})
class HandwritingCrops(Dataset):
    def __init__(self, root_dir: str, converter, img_size: tuple = (100, 64)):
        self.root = Path(root_dir)
        self.imgW, self.imgH = img_size
        self.converter = converter
        self.samples = []

        df = pd.read_csv(self.root / "merged_labels.csv", dtype={"text": str})
        for _, row in df.iterrows():
            img_path = self.root / "merged_images" / row["filename"]
            txt = row["text"]
            if not isinstance(txt, str) or not txt.strip():
                continue
            self.samples.append((img_path, txt))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.imgW, self.imgH), interpolation=cv2.INTER_AREA)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.
        return img, label

def collate_eval(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return imgs, labels

def evaluate(reader, test_loader, device="cuda"):
    reader.recognizer.eval()
    total, correct = 0, 0

    for imgs, labels in test_loader:
        imgs = imgs.to(device)

        with torch.no_grad():
            logits = reader.recognizer(imgs)
            preds_idx = logits.softmax(2).argmax(2)
            preds_size = torch.full(
                size=(preds_idx.size(0),),
                fill_value=preds_idx.size(1),
                dtype=torch.long,
                device=preds_idx.device,
            )

        if hasattr(reader.converter, "decode"):
            pred_txts, _ = reader.converter.decode(preds_idx, preds_size)
        else:
            # fallback greedy decode
            blank = 0
            idx2char = reader.converter.idx2char
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

        for gt, pr in zip(labels, pred_txts):
            cleaned_gt = re.sub(FORBIDDEN, " ", gt.translate(tbl).strip())
            cleaned_gt = re.sub(r"\s{2,}", " ", cleaned_gt)
            if cleaned_gt.replace(" ", "") == pr.replace(" ", ""):
                correct += 1
            total += 1
            print(f"GT: {cleaned_gt}\nPR: {pr}\n")

    acc = correct / total if total else 0
    wandb.log({"eval/accuracy": acc})
    print(f"\n✅ Accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_txt", default="ocr-opt.txt")
    parser.add_argument("--rec_model", required=True)
    parser.add_argument("--test_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb_name", default=None, help="W&B run name")
    parser.add_argument("--project_name", default="brainocr-eval", help="W&B project name")
    args = parser.parse_args()

    wandb.init(
        project=args.project_name,
        name=args.wandb_name if args.wandb_name else None,
        config={
            "rec_model": args.rec_model,
            "test_root": args.test_root,
            "device": args.device
        }
    )

    reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp="./assets/craft.pt",
        rec_model_ckpt_fp=args.rec_model,
        opt_fp=args.opt_txt,
        device=args.device,
    )
    reader.recognizer.to(args.device)

    test_ds = HandwritingCrops(args.test_root, reader.converter)
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False,
        num_workers=2, collate_fn=collate_eval, pin_memory=True
    )

    evaluate(reader, test_loader, device=args.device)
    wandb.finish()