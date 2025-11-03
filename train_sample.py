# train_sample.py

import argparse
import torch
import yaml
import wandb
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader
from pororo.models.brainOCR.recognition import get_recognizer
from pororo.models.brainOCR import brainocr
from pororo.models.brainOCR.brainocr import Reader
from datasets import _BaseCrops, collate_eval, collate_train, evaluate_dataset


def parse_opt(opt_txt: str) -> dict:
    """Parse option text file to dict"""
    opt2val = {}
    for ln in Path(opt_txt).read_text(encoding="utf-8").splitlines():
        if ": " not in ln:
            continue
        k, v = ln.split(": ", 1)
        try:
            opt2val[k] = yaml.safe_load(v)
        except Exception:
            opt2val[k] = v
    return opt2val


def build_recognizer(opt_txt_fp: str, device: str = "cuda"):
    """Build recognizer model from option file"""
    opt = Reader.parse_options(opt_txt_fp)

    opt["vocab"] = Reader.build_vocab(opt["character"])
    opt["vocab_size"] = len(opt["vocab"])
    opt["num_class"] = opt["vocab_size"]

    default_ckpt = Path.home() / ".pororo" / "misc" / "brainocr.pt"
    opt["rec_model_ckpt_fp"] = str(default_ckpt)
    
    opt["device"] = device

    model, converter = get_recognizer(opt)
    model.to(device)
    
    if not hasattr(converter, "decode"):
        import types
        def _decode(self, flat_idx, len_tensor):
            return self.decode_greedy(flat_idx, len_tensor)
        converter.decode = types.MethodType(_decode, converter)
    
    return model, converter, opt

def train(recognizer, converter, train_loader, epochs, lr, device="cuda"):
    """Fine-tune recognizer with CTC loss"""
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    optimizer = torch.optim.Adam(
        recognizer.parameters(), lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    recognizer.train()

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        batch_count = 0

        for step, batch in enumerate(train_loader):
            if batch is None:
                continue

            imgs, tgt, tgt_len = batch
            imgs = imgs.to(device)
            recognizer.zero_grad()

            try:
                logits = recognizer(imgs)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                input_len = torch.full(
                    (imgs.size(0),), logits.size(1), dtype=torch.long, device=device
                )
                loss = criterion(log_probs, tgt, input_len, tgt_len)

                if torch.isinf(loss) or torch.isnan(loss):
                    print(f"[skip] ep{ep} step{step} loss={loss.item()}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(recognizer.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                batch_count += 1
            except RuntimeError as e:
                print(f"[error skip] {e}")
                continue

        avg_loss = running_loss / batch_count if batch_count > 0 else float("nan")
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[epoch {ep}/{epochs}] avg_loss={avg_loss:.6f}, lr={current_lr:.2e}")
        wandb.log({"epoch": ep, "avg_loss": avg_loss, "lr": current_lr})


def save_model(recognizer, opt_dict: Dict, save_dir: Path):
    """Save checkpoint and option file"""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_fp = save_dir / "finetune_clova.pt"
    torch.save(recognizer.state_dict(), ckpt_fp)

    opt_fp = save_dir / "finetune_clova_opt.txt"
    with opt_fp.open("w", encoding="utf-8") as f:
        for k, v in opt_dict.items():
            if k == "device":
                continue
            f.write(f"{k}: {v}\n")
    print(f"Saved ckpt -> {ckpt_fp}\nSaved opt -> {opt_fp}")
    return ckpt_fp, opt_fp


def get_unique_dir(base="assets", prefix="test"):
    """Generate unique save directory"""
    import os
    os.makedirs(base, exist_ok=True)
    
    idx = 1
    while True:
        save_dir = os.path.join(base, f"{prefix}_{idx}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir
        idx += 1


def run_train(epochs, batch_size, lr, args):
    """Full training pipeline with wandb logging"""
    save_dir = get_unique_dir()
    print(f"Model save path: {save_dir}")

    wandb.init(
        project="brainocr-fine-tuning",
        name=f"{save_dir}_{epochs}_lr{lr}_bs{batch_size}_clova",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "device": args.device,
            "opt_txt": args.opt_txt,
            "train_root": args.train_root,
            "test_root": args.test_root
        }
    )

    rec, converter, opt_dict = build_recognizer(args.opt_txt, device=args.device)

    train_set = _BaseCrops(
        csv_fp=Path(args.train_root) / "train_labels.csv",
        img_dir=Path(args.train_root) / "merged_images",
        img_size=(256, 64),
        converter=converter,
        for_train=True,
    )
    train_set.preload_for_stats()
    train_set.print_filter_report()

    test_set = _BaseCrops(
        csv_fp=Path(args.test_root) / "test_labels.csv",
        img_dir=Path(args.test_root) / "merged_images",
        img_size=(256, 64),
        for_train=False,
    )
    test_set.preload_for_stats()
    test_set.print_filter_report()

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_train, drop_last=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_eval, pin_memory=True
    )

    train(rec, converter, train_loader, epochs=epochs, lr=lr, device=args.device)

    ckpt_fp, opt_fp = save_model(rec, opt_dict, Path(save_dir))

    reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp="/home/heven/SKKUOCR/assets/craft.pt",
        rec_model_ckpt_fp=str(ckpt_fp),
        opt_fp=str(opt_fp),
        device=args.device,
    )
    reader.recognizer.to(args.device)

    evaluate_dataset(reader, test_loader, device=args.device, save_csv="assets/train_pred.csv")

    wandb.finish()


def run_test(model_ckpt_fp, opt_fp, test_root, device="cuda"):
    """Test only mode with saved checkpoint"""
    reader = brainocr.Reader(
        lang="ko",
        det_model_ckpt_fp="/home/heven/SKKUOCR/assets/craft.pt",
        rec_model_ckpt_fp=str(model_ckpt_fp),
        opt_fp=str(opt_fp),
        device=device,
    )
    reader.recognizer.to(device)

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

    evaluate_dataset(reader, test_loader, device=device, save_csv="assets/test_pred.csv")

    print("Test complete. Results saved to assets/test_pred.csv")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_txt", default="ocr-opt.txt")
    parser.add_argument("--train_root", default="CLOVA_V3_train")
    parser.add_argument("--test_root", default="CLOVA_V2_test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="assets")
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # train
    run_train(args.epochs, args.batch, args.lr, args)

    # test
    run_test(
        model_ckpt_fp="assets/test_20/finetune_recognizer.pt",
        opt_fp="assets/test_20/finetune_recognizer_opt.txt",
        test_root="CLOVA_V2_test",
        device="cuda"
    )