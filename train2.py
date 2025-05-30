"""
Finetune Pororo brainOCR recognizer (CTC-based)
────────────────────────────────────────────────
· cuDNN OFF  (grid_sample 경고 회피)
· 라벨 clean  → 개행 제거·특수 기호 치환
· 동적 vocab  → 학습 데이터에서 새 문자 추가
· bbox 단위 crop 학습 + PAD
· CTC 길이 제약 필터  (tg_len==0 or tg_len>T)
· 첫 배치 / val 단계 rich 디버그 & 오답 이미지 저장
"""
# ───── 0. 공용 import ───────────────────────────────────────────────────────
import os, random, json, math
from collections import Counter

import cv2, numpy as np
from PIL import Image
from tqdm import tqdm

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pororo import Pororo
from pororo.models.brainOCR.recognition import CTCLabelConverter

# ─── 1. 시드 고정 ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# ─── 2. 라벨 클린 ─────────────────────────────────────────────────────────
CHAR_MAP = {
    'λ': 'lambda', '√': 'sqrt', '≤': '<=', '≥': '>=',
    '‖': '||', '⋯': '...', '∈': 'in', '∑': 'sum', 'β': 'beta', 'ω': 'omega'
}
def clean_label(txt: str) -> str:
    txt = txt.replace('\n', ' ').strip()
    txt = txt.translate(str.maketrans(CHAR_MAP))
    return " ".join(txt.split())

# ─── 3. PAD │ Dataset (이전과 동일) ────────────────────────────────────────
class NormalizePAD:
    def __init__(self, imgH=64, imgW=256):
        self.imgH, self.imgW = imgH, imgW
    def __call__(self, img: Image.Image):
        w, h = img.size
        new_w = self.imgW if h*(w/h) > self.imgW else int(h*(w/h))
        img   = img.resize((new_w, self.imgH), Image.BICUBIC)
        pad   = Image.new("L", (self.imgW, self.imgH), 0)
        pad.paste(img, (0,0))
        pad = (np.array(pad).astype(np.float32)/255. - .5)/.5
        return torch.from_numpy(pad).unsqueeze(0)

class OCRFinetuningDataset(Dataset):
    """
    image_root: ./train or ./test
    label_root: ./text_data/train or ./text_data/test
    하위에 image / notice / handwriting 폴더
    """
    def __init__(self, image_root, label_root, transform=None, max_samples=-1, base_conv=None):
        self.transform = transform
        self.samples = []
        self.excluded_samples = []  # 제외된 샘플 기록

        # base_conv에서 charset 가져오기
        if base_conv is not None:
            self.charset = set(_charset(base_conv))
        else:
            self.charset = None

        # cats = [d for d in os.listdir(image_root)
        #         if os.path.isdir(os.path.join(image_root, d))]
        cats = ['image', 'notice']
        print(f"[INFO] {os.path.basename(image_root)} 카테고리: {cats}")

        for cat in cats:
            img_cat = os.path.join(image_root, cat)
            lbl_cat = os.path.join(label_root, cat)

            imgs = [f for f in os.listdir(img_cat)
                    if f.lower().endswith((".png",".jpg",".jpeg"))]
            if 0 < max_samples < len(imgs): 
                imgs = random.sample(imgs, max_samples)

            for fn in imgs:
                img_fp = os.path.join(img_cat, fn)
                base = os.path.splitext(fn)[0]

                txt_fp = os.path.join(lbl_cat, f"{base}_label.txt")
                json_fp = os.path.join(lbl_cat, f"{base}_label.json")

                # txt 1줄
                if os.path.exists(txt_fp):
                    lbl = clean_label(open(txt_fp, encoding="utf-8").read())
                    if self._is_valid_label(lbl):
                        self.samples.append(dict(img=img_fp, label=lbl,
                                                bbox=-1, xyxy=None, cat=cat))
                    else:
                        self.excluded_samples.append((img_fp, lbl, "Invalid chars in txt"))

                # json 다중 bbox
                if os.path.exists(json_fp):
                    for ent in json.load(open(json_fp, encoding="utf-8")):
                        lbl = clean_label(ent["corrected_text"])
                        bidx = ent["bbox_index"]
                        xs = [v["x"] for v in ent["bbox"]["vertices"]]
                        ys = [v["y"] for v in ent["bbox"]["vertices"]]
                        x1, x2 = max(min(xs), 0), max(xs)
                        y1, y2 = max(min(ys), 0), max(ys)
                        if self._is_valid_label(lbl):
                            self.samples.append(dict(img=img_fp, label=lbl,
                                                    bbox=bidx, xyxy=(x1, y1, x2, y2),
                                                    cat=cat))
                        else:
                            self.excluded_samples.append((img_fp, lbl, f"Invalid chars in json[bbox{bidx}]"))

        print(f"[INFO] {os.path.basename(image_root)} 샘플 수: {len(self.samples)}")
        if self.excluded_samples:
            print(f"[INFO] 제외된 샘플 수: {len(self.excluded_samples)}")
            for img_fp, lbl, reason in self.excluded_samples[:5]:  # 최대 5개만 출력
                print(f"[EXCLUDED] {img_fp} :: {reason} (label: {lbl})")
                
    def _is_valid_label(self, label):
        """라벨이 charset에 포함된 문자들로만 구성되어 있는지 확인"""
        if self.charset is None:
            return True  
        return all(char in self.charset for char in label)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = cv2.cvtColor(cv2.imread(s["img"]), cv2.COLOR_BGR2RGB)

        if s["xyxy"] is not None:                    # bbox crop
            x1,y1,x2,y2 = s["xyxy"]
            crop = src[y1:y2, x1:x2]
            if crop.size==0: crop = src
        else:
            crop = src

        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
        img = self.transform(img) if self.transform else img
        return dict(image=img,label=s["label"],
                    path=s["img"],bbox=s["bbox"],cat=s["cat"])

# ───── 4. recognizer & converter ───────────────────────────────────────────
def get_recognizer():
    reader = Pororo(task="ocr", lang="ko", model="brainocr")._model
    return reader.recognizer, reader.converter

# ─── 5. vocab (★★ base 그대로 ★★) ─────────────────────────────────────────
def _charset(conv) -> str:
    if hasattr(conv, "character"):   return conv.character
    if hasattr(conv, "idx2char"):
        d = conv.idx2char
        return "".join(d[k] for k in sorted(d) if k) if isinstance(d, dict) else "".join(d[1:])
    if hasattr(conv, "characters"):
        ch = conv.characters
        return "".join(ch) if isinstance(ch, (list, tuple)) else ch
    raise AttributeError

def build_converter(train_ds, base_conv):
    """Pororo 기본 charset만 사용 (새 문자 추가하지 않음)"""
    charset = _charset(base_conv)
    
    # 학습 데이터의 모든 문자 확인 (디버깅용)
    all_chars = set("".join(s["label"] for s in train_ds.samples))
    print(f"All characters in dataset: {''.join(sorted(all_chars))}")
    
    # 제외된 문자 확인
    excluded_chars = all_chars - set(charset)
    if excluded_chars:
        print(f"[INFO] 제외된 문자: {''.join(sorted(excluded_chars))}")
    
    print(f"[INFO] vocab 길이: {len(charset)} (추가 0자)")
    return CTCLabelConverter(["[blank]"] + list(charset))
def _charset_from_conv(conv):
    if hasattr(conv, "character"):             # str
        return set(conv.character)
    if hasattr(conv, "characters"):            # list[str] or str
        chars = conv.characters
        return set(chars) if isinstance(chars, str) else set("".join(chars))
    if hasattr(conv, "idx2char"):              # dict or list
        idx2char = conv.idx2char
        if isinstance(idx2char, dict):
            return set(idx2char[k] for k in idx2char if k)  # k==0 → blank
        return set(idx2char[1:])               # idx0 == blank
    raise AttributeError("unknown converter format")

# ───── 6. 학습 ──────────────────────────────────────────────────────────────
def train(recog, conv, train_ds, test_ds, out_dir,
          epochs=5, device="cuda", dbg_n=5, save_bad=True):

    recog.to(device)
    opt = optim.Adam(recog.parameters(), lr=3e-5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    os.makedirs(out_dir, exist_ok=True)
    if save_bad: os.makedirs(f"{out_dir}/mis_pred", exist_ok=True)

    mkload = lambda ds,sh:DataLoader(ds,8,shuffle=sh,num_workers=2,
                                     pin_memory=(device=="cuda"))

    for ep in range(1, epochs+1):
        recog.train(); run=0.0
        bar=tqdm(mkload(train_ds,True),desc=f"E{ep}[train]")
        for b,bt in enumerate(bar):
            imgs   = bt["image"].to(device)
            labels = bt["label"]
            
            charset = _charset_from_conv(conv)
            
            good_idx = [i for i, t in enumerate(labels) if all(ch in charset for ch in t)]

            if not good_idx:
                continue
            imgs   = imgs[good_idx]
            labels = [labels[i] for i in good_idx]
            
            logits = recog(imgs)
            targets, len_all = conv.encode(bt["label"])

            # 길이 필터 (tg_len>logits_T → 오류, tg_len==0 → 빈라벨)
            # keep = (len_all>0) & (len_all <= logits.size(1)-2)
            # ▶ **길이 == T 인 경우도 제외** (PyTorch CTCLoss 의 NaN 버그 회피)
            keep = (len_all>0) & (len_all < logits.size(1))
            
            if not keep.any(): continue
            idx  = keep.nonzero(as_tuple=True)[0]
            tg,len_= conv.encode([bt["label"][i] for i in idx])

            logit = logits[idx]; logp=logit.log_softmax(2).permute(1,0,2)
            inp_len=torch.full((logit.size(0),),logit.size(1),dtype=torch.long)
            
            loss=ctc(logp.cpu(),tg,inp_len,len_)
            
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                continue

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(recog.parameters(), 5.0)
            opt.step()
            run+=loss.item(); bar.set_postfix(loss=f"{loss.item():.3f}")

            # 첫 배치 디버그
            if ep==1 and b==0:
                pidx=logit.softmax(2).max(2)[1]
                pred=conv.decode_greedy(pidx.view(-1).cpu(),
                        torch.IntTensor([logit.size(1)]*logit.size(0)))
                for i, j in enumerate(idx):
                    pr = pred[i]
                    gt = bt["label"][j]
                    path = bt["path"][j]
                    bbox = bt["bbox"][j]
                    tag  = f"{path}[bbox{bbox}]" if bbox != -1 else path
                    print(f"[DBG] {tag} :: pred ▶ {pr}\n                    gt   ▶ {gt}")
                    if i + 1 == dbg_n:
                        break

        # ─ val ─
        recog.eval(); cor=tot=0
        val_dbg_shown = 0
        with torch.no_grad():
            for bt in mkload(test_ds,False):
                imgs=bt["image"].to(device); logit=recog(imgs)
                pidx=logit.softmax(2).max(2)[1]
                pred=conv.decode_greedy(pidx.view(-1).cpu(),
                      torch.IntTensor([logit.size(1)]*imgs.size(0)))
                for im,pr,gt,pt,bx,ct in zip(imgs,pred,bt["label"],
                                              bt["path"],bt["bbox"],bt["cat"]):
                    tot+=1
                    if pr==gt: cor+=1
                    elif val_dbg_shown < dbg_n:      # 🔸추가
                        tag = f"{pt}[bbox{bx}]" if bx != -1 else pt
                        print(f"[VAL-DBG] {tag} :: pred ▶ {pr}\n"
                              f"                      gt   ▶ {gt}")
                        val_dbg_shown += 1
                    elif save_bad:
                        name,_=os.path.splitext(os.path.basename(pt))
                        bx_tag=f"_bbox{bx}" if bx!=-1 else ""
                        fn=f"{out_dir}/mis_pred/ep{ep}_{ct}_{name}{bx_tag}.png"
                        cv2.imwrite(fn,((im[0].cpu().numpy()*0.5+0.5)*255).astype(np.uint8))
                        with open(fn.replace(".png",".txt"),"w",encoding="utf-8") as f:
                            f.write(f"pred: {pr}\nlabel: {gt}\nfile : {pt}")
        print(f"\nE{ep}: mean_loss={run/len(train_ds):.4f}  val_acc={cor/tot:.4f}")
        torch.save(recog.state_dict(), f"{out_dir}/epoch{ep}.pth")

def _num_classes(conv):
    if hasattr(conv,"character"):   return len(conv.character)+1
    if hasattr(conv,"characters"):  return len(conv.characters)+1
    if hasattr(conv,"idx2char"):    return len(conv.idx2char)
    raise AttributeError

# ───── 7. main ─────────────────────────────────────────────────────────────
def main():
    img_train, img_test = "./train", "./test"
    lbl_train, lbl_test = "./text_data/train", "./text_data/test"
    out_dir = "finetune_results"

    tfm = transforms.Compose([NormalizePAD(64, 256)])  # PAD 포함

    # recognizer와 base_conv를 먼저 가져와 charset을 데이터셋에 전달
    recog, base_conv = get_recognizer()
    train_ds = OCRFinetuningDataset(img_train, lbl_train, tfm, max_samples=10, base_conv=base_conv)
    test_ds = OCRFinetuningDataset(img_test, lbl_test, tfm, max_samples=10, base_conv=base_conv)

    conv = build_converter(train_ds, base_conv)

    old_fc, old_cls = recog.Prediction, recog.Prediction.out_features
    new_cls = _num_classes(conv)
    if new_cls != old_cls:
        print(f"[INFO] expand classifier {old_cls}→{new_cls}")
        new_fc = nn.Linear(old_fc.in_features, new_cls, bias=True)
        with torch.no_grad():
            copy = min(old_cls, new_cls)
            new_fc.weight[:copy].copy_(old_fc.weight[:copy])
            new_fc.bias[:copy].copy_(old_fc.bias[:copy])
            if new_cls > old_cls:
                nn.init.xavier_uniform_(new_fc.weight[old_cls:])
                nn.init.zeros_(new_fc.bias[old_cls:])
        recog.Prediction = new_fc

    # 두 그룹 learning-rate
    old_params, new_params = [], []
    for n, p in recog.named_parameters():
        if "Prediction" in n and p.shape[0] == new_cls:
            new_params.append(p)  # 새 클래스 weight
        else:
            old_params.append(p)
    optimizer = optim.Adam([
        {'params': old_params, 'lr': 3e-5},
        {'params': new_params, 'lr': 3e-4},
    ])

    train(recog, conv, train_ds, test_ds, "finetune_results",
          epochs=100, device="cuda" if torch.cuda.is_available() else "cpu",
          dbg_n=5, save_bad=True)

if __name__ == "__main__":
    main()