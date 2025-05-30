"""
Finetune Pororo brainOCR recognizer (CTC-based)
────────────────────────────────────────────────
· cuDNN OFF (grid_sample 경고 회피)
· 라벨 clean → 개행 제거·특수 기호 치환
· 동적 vocab → 학습 데이터에서 새 문자 추가 (현재 요청에 따라 비활성화)
· bbox 단위 crop 학습 + PAD
· CTC 길이 제약 필터 (tg_len==0 or tg_len>T)
· 첫 배치 / val 단계 rich 디버그 & 오답 이미지 저장
· CRAFT 모델을 사용하여 이미지에서 텍스트 바운딩 박스를 동적으로 탐지 및 크롭
"""
# ───── 0. 공용 import ───────────────────────────────────────────────────────
import os, random, json, math
from collections import Counter, OrderedDict

import cv2, numpy as np
from PIL import Image
from tqdm import tqdm

import torch
torch.backends.cudnn.enabled = False # grid_sample 경고 회피
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pororo import Pororo
from pororo.models.brainOCR.recognition import CTCLabelConverter

# Pororo 내부 CRAFT 및 Recognition 모델 파일 경로 설정
# 이 경로는 당신의 ~/.pororo/misc/ 폴더를 가리켜야 합니다.
# 예시: "/home/parkprogrammer/.pororo/misc"
# !!! 중요: 이 변수를 당신의 실제 Pororo 모델 저장 경로로 수정하세요!
PORORO_MISC_DIR = os.path.expanduser("~/.pororo/misc")
DET_MODEL_CKPT_FP = os.path.join(PORORO_MISC_DIR, "craft.pt")
REC_MODEL_CKPT_FP = os.path.join(PORORO_MISC_DIR, "brainocr.pt")
OPT_FP = os.path.join(PORORO_MISC_DIR, "ocr-opt.txt")

# --- CRAFT Detection 관련 함수 (detection.py에서 직접 가져옴) ---
# 필요한 경우 pororo.models.brainOCR.craft 및 imgproc 도 복사해야 할 수 있습니다.
# 여기서는 편의를 위해 해당 파일들이 같은 디렉토리 또는 Python path에 있다고 가정합니다.
# 만약 오류 발생 시, 해당 파일들을 직접 이 스크립트와 같은 폴더에 복사하거나,
# 경로를 정확히 지정하여 임포트해야 합니다.
try:
    from pororo.models.brainOCR.craft import CRAFT
    from pororo.models.brainOCR.craft_utils import adjust_result_coordinates, get_det_boxes
    from pororo.models.brainOCR.imgproc import normalize_mean_variance, resize_aspect_ratio
    from pororo.models.brainOCR.utils import get_image_list, group_text_box, reformat_input
except ImportError as e:
    print(f"Error importing Pororo internal CRAFT modules: {e}")
    print("Please ensure pororo.models.brainOCR.craft, craft_utils, imgproc, utils are accessible.")
    print("You might need to copy them to your project directory or adjust PYTHONPATH.")
    # 오류 방지를 위한 더미 클래스/함수 (실제 사용 시에는 구현 필요)
    class CRAFT(nn.Module):
        def __init__(self): super().__init__(); print("Dummy CRAFT class")
        def forward(self, x): return None, None
    def adjust_result_coordinates(*args, **kwargs): return []
    def get_det_boxes(*args, **kwargs): return [], []
    def normalize_mean_variance(*args, **kwargs): return np.zeros((1,1,3))
    def resize_aspect_ratio(*args, **kwargs): return np.zeros((1,1,3)), 1.0, (1,1)
    def get_image_list(*args, **kwargs): return [], []
    def group_text_box(*args, **kwargs): return [], []
    def reformat_input(*args, **kwargs): return None, np.zeros((1,1,3))


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(image: np.ndarray, net, opt2val: dict):
    canvas_size = opt2val["canvas_size"]
    mag_ratio = opt2val["mag_ratio"]
    text_threshold = opt2val["text_threshold"]
    link_threshold = opt2val["link_threshold"]
    low_text = opt2val["low_text"]
    device = opt2val["device"]

    # --- 추가된 코드: 이미지가 1채널인 경우 3채널로 변환 ---
    if len(image.shape) == 2: # 그레이스케일 이미지인 경우 (H, W)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # BGR로 변환하여 (H, W, 3) 형태로 만듦
    elif len(image.shape) == 3 and image.shape[2] == 1: # 1채널이지만 (H, W, 1) 형태인 경우
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # BGR로 변환하여 (H, W, 3) 형태로 만듦
    # --- 추가된 코드 끝 ---

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalize_mean_variance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = get_det_boxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
    )

    # coordinate adjustment
    boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
    polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys


def get_detector(det_model_ckpt_fp: str, device: str = "cpu"):
    net = CRAFT()

    net.load_state_dict(
        copy_state_dict(torch.load(det_model_ckpt_fp, map_location=device)))
    if device == "cuda" and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)

    # cudnn.benchmark = True

    net.eval()
    return net


def get_textbox(detector, image: np.ndarray, opt2val: dict):
    bboxes, polys = test_net(image, detector, opt2val)
    result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return result

# --- CRAFT Detection 관련 함수 끝 ---


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

# ─── 3. PAD │ Dataset (수정된 부분) ────────────────────────────────────────
class NormalizePAD:
    def __init__(self, imgH=64, imgW=256):
        self.imgH, self.imgW = imgH, imgW
    def __call__(self, img: Image.Image):
        w, h = img.size
        if h == 0:
            aspect_ratio = 1.0
        else:
            aspect_ratio = w / h

        new_w = int(self.imgH * aspect_ratio)
        if new_w > self.imgW:
            new_w = self.imgW

        img = img.resize((new_w, self.imgH), Image.BICUBIC)

        pad = Image.new("L", (self.imgW, self.imgH), 0)
        pad.paste(img, (0, 0)) # 좌측 상단 정렬

        pad = (np.array(pad).astype(np.float32)/255. - .5)/.5
        return torch.from_numpy(pad).unsqueeze(0)

class OCRFinetuningDataset(Dataset):
    def __init__(self, image_root, label_root, transform=None, max_samples=-1, base_conv=None,
                 pororo_detector=None, pororo_opt2val=None):
        self.transform = transform
        self.samples = []
        self.excluded_samples = []
        self.pororo_detector = pororo_detector # Pororo Reader에서 가져온 detector 인스턴스
        self.pororo_opt2val = pororo_opt2val # Pororo Reader의 opt2val 딕셔너리

        if base_conv is not None:
            self.charset = set(_charset(base_conv))
        else:
            self.charset = None

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

                if os.path.exists(json_fp):
                    for ent in json.load(open(json_fp, encoding="utf-8")):
                        lbl = clean_label(ent["corrected_text"])
                        bidx = ent["bbox_index"]
                        if self._is_valid_label(lbl):
                            self.samples.append(dict(img=img_fp, label=lbl,
                                                     bbox=bidx, xyxy=None,
                                                     cat=cat))
                        else:
                            self.excluded_samples.append((img_fp, lbl, f"Invalid chars in json[bbox{bidx}]"))
                elif os.path.exists(txt_fp):
                    lbl = clean_label(open(txt_fp, encoding="utf-8").read())
                    if self._is_valid_label(lbl):
                        self.samples.append(dict(img=img_fp, label=lbl,
                                                 bbox=-1, xyxy=None, cat=cat))
                    else:
                        self.excluded_samples.append((img_fp, lbl, "Invalid chars in txt"))

        print(f"[INFO] {os.path.basename(image_root)} 샘플 수: {len(self.samples)}")
        if self.excluded_samples:
            print(f"[INFO] 제외된 샘플 수: {len(self.excluded_samples)}")
            for img_fp, lbl, reason in self.excluded_samples[:5]:
                print(f"[EXCLUDED] {img_fp} :: {reason} (label: {lbl})")

    def _is_valid_label(self, label):
        if self.charset is None:
            return True
        return all(char in self.charset for char in label)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src_np = cv2.cvtColor(cv2.imread(s["img"]), cv2.COLOR_BGR2RGB) # 원본 이미지 (RGB)

        # 1. Pororo의 reformat_input을 사용하여 CRAFT 입력 이미지 생성
        _, img_for_detect = reformat_input(src_np)

        # 2. Pororo의 get_textbox (CRAFT 모델 추론 및 후처리)
        text_box_list = get_textbox(self.pororo_detector, img_for_detect, self.pororo_opt2val)

        # 3. Pororo의 group_text_box (바운딩 박스 그룹화)
        horizontal_list, free_list = group_text_box(
            text_box_list,
            self.pororo_opt2val["slope_ths"],
            self.pororo_opt2val["ycenter_ths"],
            self.pororo_opt2val["height_ths"],
            self.pororo_opt2val["width_ths"],
            self.pororo_opt2val["add_margin"],
        )

        # 4. Pororo의 get_image_list (그룹화된 박스에서 이미지 크롭)
        # Note: opt2val에서 imgH를 가져오는 것이 아니라, Reader 객체의 __call__에서 설정되는 imgH를 사용해야 합니다.
        # Pororo Reader 클래스의 __init__ 에서는 imgH가 `opt2val`에 직접 포함되지 않음.
        # 대신 `Reader`의 `__call__` 메서드에서 `opt2val['imgH'] = 64`처럼 설정될 수 있음.
        # 우리는 학습 데이터셋에서 이미지를 리사이즈하는 기준을 `NormalizePAD`에서 정했으므로,
        # CRAFT의 `get_image_list`에 넘겨주는 `model_height`는 `NormalizePAD`의 `imgH`와 일치시키는 것이 좋습니다.
        # 여기서는 기본적으로 `NormalizePAD`의 `imgH`인 64를 사용합니다.
        # 만약 `self.pororo_opt2val`에 `imgH`가 있다면 그것을 사용하고, 없다면 기본값 64를 사용.
        imgH = self.pororo_opt2val.get("imgH", 64)

        image_list, _ = get_image_list(
            horizontal_list,
            free_list,
            img_for_detect, # 그레이스케일 np.ndarray 이미지 전달
            model_height=imgH,
        )

        selected_crop_img = None
        # --- 라벨에 가장 적합한 크롭 이미지 선택 로직 ---
        # 여기서는 가장 큰 바운딩 박스에 해당하는 크롭 이미지를 선택합니다.
        if image_list:
            max_area = -1
            best_crop = None
            for bbox_coords, crop_img_np in image_list:
                x_coords = [p[0] for p in bbox_coords]
                y_coords = [p[1] for p in bbox_coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                area = (x_max - x_min) * (y_max - y_min)
                if area > max_area:
                    max_area = area
                    best_crop = crop_img_np

            selected_crop_img = best_crop

            if selected_crop_img is None or selected_crop_img.size == 0 or selected_crop_img.shape[0] == 0 or selected_crop_img.shape[1] == 0:
                selected_crop_img = src_np # 유효하지 않은 크롭 방지
                print(f"[WARN] Invalid CRAFT-detected crop for {s['img']}, using full image (area problem).")
        else:
            selected_crop_img = src_np # CRAFT가 아무것도 탐지하지 못하면 원본 이미지 사용
            print(f"[WARN] CRAFT detected no text for {s['img']}, using full image.")

        img_pil = Image.fromarray(selected_crop_img).convert("L")
        img = self.transform(img_pil) if self.transform else img

        return dict(image=img,label=s["label"],
                    path=s["img"],bbox=s["bbox"],cat=s["cat"])

# ───── 4. recognizer & converter ───────────────────────────────────────────
# Pororo Reader의 내부 Reader 클래스 임포트
try:
    from pororo.models.brainOCR import Reader as PororoReader
except ImportError as e:
    print(f"Error importing Pororo internal Reader: {e}")
    print("Please ensure Pororo is installed correctly or copy the necessary files.")
    raise

def get_recognizer_and_options(det_model_ckpt_fp: str, rec_model_ckpt_fp: str, opt_fp: str, device: str):
    # Pororo Reader 인스턴스 생성 (이 과정에서 det/rec 모델이 로드됩니다)
    # Reader의 생성자 인수에 맞게 실제 파일 경로를 전달합니다.
    reader_instance = PororoReader(
        lang="ko",
        det_model_ckpt_fp=det_model_ckpt_fp,
        rec_model_ckpt_fp=rec_model_ckpt_fp,
        opt_fp=opt_fp,
        device=device
    )

    # Reader의 __call__ 메서드를 호출하여 opt2val에 CRAFT 파라미터들을 업데이트
    # 어떤 이미지든 임의로 하나 넣어주면 됩니다. 여기서는 더미 이미지를 사용합니다.
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8) # 어떤 이미지든 상관 없음
    # __call__을 호출하면 모든 파라미터들이 self.opt2val에 업데이트됩니다.
    reader_instance(dummy_image,
                    # CRAFT 관련 파라미터는 `brainocr.py`의 `__call__`에서 기본값을 가져옵니다.
                    # 여기서는 명시적으로 몇 가지를 설정하여 `opt2val`에 포함되도록 합니다.
                    canvas_size=1280, # 예시 값. Pororo 기본값과 일치하거나 조정 가능
                    mag_ratio=1.5,
                    text_threshold=0.7,
                    low_text=0.4,
                    link_threshold=0.4,
                    slope_ths=0.1,
                    ycenter_ths=0.5,
                    height_ths=0.5,
                    width_ths=0.5,
                    add_margin=0.1,
                    min_size=20 # Reader의 min_size 파라미터
                    )


    return reader_instance.detector, reader_instance.recognizer, reader_instance.converter, reader_instance.opt2val

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

    all_chars_in_dataset = set("".join(s["label"] for s in train_ds.samples))
    print(f"All characters found in dataset labels: {''.join(sorted(all_chars_in_dataset))}")

    excluded_chars_from_base_vocab = all_chars_in_dataset - set(charset)
    if excluded_chars_from_base_vocab:
        print(f"[WARN] Pororo 기본 vocab에 없는 문자: {''.join(sorted(excluded_chars_from_base_vocab))}")
        print(f"[WARN] 이 문자들은 모델이 예측할 수 없으며, 해당 라벨은 학습에서 제외될 수 있습니다.")

    print(f"[INFO] vocab 길이: {len(charset)} (Pororo 기본 어휘 사용)")
    return CTCLabelConverter(["[blank]"] + list(charset))

def _charset_from_conv(conv):
    if hasattr(conv,"character"):    return set(conv.character)
    if hasattr(conv,"characters"):
        chars = conv.characters
        return set(chars) if isinstance(chars, str) else set("".join(chars))
    if hasattr(conv,"idx2char"):
        idx2char = conv.idx2char
        if isinstance(idx2char, dict):
            return set(idx2char[k] for k in idx2char if k) # k==0 -> blank
        return set(idx2char[1:])     # idx0 == blank
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
            labels_filtered = [labels[i] for i in good_idx]

            logits = recog(imgs)
            targets, len_all = conv.encode(labels_filtered)

            keep = (len_all > 0) & (len_all <= logits.size(1))

            if not keep.any(): continue
            idx_for_ctc   = keep.nonzero(as_tuple=True)[0]
            tg,len_ = conv.encode([labels_filtered[i] for i in idx_for_ctc])

            logit = logits[idx_for_ctc]; logp=logit.log_softmax(2).permute(1,0,2)
            inp_len=torch.full((logit.size(0),),logit.size(1),dtype=torch.long)

            loss=ctc(logp.cpu(),tg,inp_len,len_)

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f"[WARN] Epoch {ep}, Batch {b}: Loss is NaN/Inf, skipping. Labels: {labels_filtered}")
                continue

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(recog.parameters(), 5.0)
            opt.step()
            run+=loss.item(); bar.set_postfix(loss=f"{loss.item():.3f}")

            if ep==1 and b==0:
                pidx=logit.softmax(2).max(2)[1]
                pred=conv.decode_greedy(pidx.view(-1).cpu(),
                                 torch.IntTensor([logit.size(1)]*logit.size(0)))
                print("[DBG] First batch predictions:")
                for i, original_j in enumerate(good_idx):
                    if i >= len(pred): continue
                    pr = pred[i]
                    gt = labels[original_j]
                    path = bt["path"][original_j]
                    bbox = bt["bbox"][original_j]
                    tag  = f"{path}[bbox{bbox}]" if bbox != -1 else path
                    print(f"[DBG] {tag} :: pred ▶ {pr}\n           gt   ▶ {gt}")
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
                    elif val_dbg_shown < dbg_n:
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
        print(f"\nE{ep}: mean_loss={run/len(train_ds):.4f}   val_acc={cor/tot:.4f}")
        torch.save(recog.state_dict(), f"{out_dir}/epoch{ep}.pth")

def _num_classes(conv):
    if hasattr(conv,"character"):   return len(conv.character)+1
    if hasattr(conv,"characters"):  return len(conv.characters)+1
    if hasattr(conv,"idx2char"):    return len(conv.idx2char)
    raise AttributeError

# ───── 7. main (수정된 부분) ─────────────────────────────────────────────────────────────
def main():
    img_train, img_test = "./train", "./test"
    lbl_train, lbl_test = "./text_data/train", "./text_data/test"
    out_dir = "finetune_results"

    tfm = transforms.Compose([NormalizePAD(64, 256)]) # PAD 포함

    PORORO_MISC_DIR = os.path.expanduser("~/.pororo/misc")
    DET_MODEL_CKPT_FP = os.path.join(PORORO_MISC_DIR, "craft.pt")
    REC_MODEL_CKPT_FP = os.path.join(PORORO_MISC_DIR, "brainocr.pt")
    OPT_FP = os.path.join(PORORO_MISC_DIR, "ocr-opt.txt")

    if not all(os.path.exists(p) for p in [DET_MODEL_CKPT_FP, REC_MODEL_CKPT_FP, OPT_FP]):
        print(f"Error: Pororo internal model/option files not found.")
        print(f"Check the PORORO_MISC_DIR: '{PORORO_MISC_DIR}' and ensure paths below are correct:")
        print(f"Det: {DET_MODEL_CKPT_FP}")
        print(f"Rec: {REC_MODEL_CKPT_FP}")
        print(f"Opt: {OPT_FP}")
        print("Please ensure Pororo models are downloaded or manually copied to this location.")
        print("You can try running `Pororo(task='ocr', lang='ko', model='brainocr', force_download=True)` once to download them.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. PororoReader 인스턴스를 통해 모든 모델과 옵션 가져오기
    # CRAFT Detector는 get_detector를 통해 직접 로드합니다.
    pororo_detector = get_detector(DET_MODEL_CKPT_FP, device)

    # PororoReader 인스턴스를 생성하여 recognizer, converter, 그리고 전체 opt2val을 가져옵니다.
    # 이때, Reader의 __call__ 메서드를 한 번 호출하여 opt2val을 최신 CRAFT 파라미터들로 업데이트합니다.
    temp_reader = PororoReader(
        lang="ko",
        det_model_ckpt_fp=DET_MODEL_CKPT_FP,
        rec_model_ckpt_fp=REC_MODEL_CKPT_FP,
        opt_fp=OPT_FP,
        device=device
    )

    # __call__ 메서드를 호출하여 내부 opt2val에 CRAFT 관련 파라미터를 채웁니다.
    # 어떤 이미지든 상관 없으며, 결과는 사용하지 않습니다.
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    temp_reader(dummy_image) # 이 호출을 통해 temp_reader.opt2val이 CRAFT 관련 옵션들로 업데이트됩니다.

    recog = temp_reader.recognizer
    base_conv = temp_reader.converter
    pororo_opt2val = temp_reader.opt2val # 이제 이 딕셔너리에 필요한 모든 CRAFT 옵션이 포함됩니다.
    
    # device는 이미 opt2val 내에 설정되어 있지만, 명시적으로 재확인
    pororo_opt2val["device"] = device


    # 2. OCRFinetuningDataset에 Pororo Detector와 옵션 전달
    train_ds = OCRFinetuningDataset(img_train, lbl_train, tfm, max_samples=-1,
                                    base_conv=base_conv,
                                    pororo_detector=pororo_detector,
                                    pororo_opt2val=pororo_opt2val)

    test_ds = OCRFinetuningDataset(img_test, lbl_test, tfm, max_samples=-1,
                                   base_conv=base_conv,
                                   pororo_detector=pororo_detector,
                                   pororo_opt2val=pororo_opt2val)

    # converter는 Pororo 기본 어휘를 그대로 사용하도록 build_converter를 호출
    conv = build_converter(train_ds, base_conv)

    # 모든 파라미터를 동일한 학습률로 미세 조정
    optimizer = optim.Adam(recog.parameters(), lr=3e-5)

    train(recog, conv, train_ds, test_ds, out_dir,
          epochs=100, device=device,
          dbg_n=5, save_bad=True)

if __name__ == "__main__":
    
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()