import os, json, cv2, random, warnings, numpy as np
import shutil, gc
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pororo import Pororo
from PIL import Image
import matplotlib.pyplot as plt # Added for plotting

warnings.filterwarnings("ignore")

# --- 0. 경로 및 학습 설정 (OOM 방지 최적화) ---
IMG_ROOT  = "train"
JSON_ROOT = "text_data/train"
TMP_CROP  = "crops_tmp"

EPOCHS = 30
BATCH = 1 
ACCUMULATION_STEPS = 4 
LR = 1e-5
SAMPLE_LIMIT = 40 
# OOM 방지 설정
CACHE_CLEAR_FREQUENCY = 5 
ENABLE_CHECKPOINTING = True  
print("Pororo OCR 모델 로딩 중...")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # 더 작은 청크
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 디버깅용

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

try:
    ocr_model = Pororo(task="ocr", lang="ko", model="brainocr")
    rdr = ocr_model._model
    
    feat_extractor = nn.Sequential(
        rdr.recognizer.Transformation,
        rdr.recognizer.FeatureExtraction
    ).eval()
    
    # 체크포인팅 활성화 (메모리 절약) - checkpoint_wrapper 대신 기본 checkpoint 함수 사용
    # 이전 코드: feat_extractor = torch.utils.checkpoint.checkpoint_wrapper(feat_extractor)
    # 수정: checkpoint는 forward pass에서 적용할 것임
    
    feat_extractor = feat_extractor.cuda()
    
    for p in feat_extractor.parameters():
        p.requires_grad = False
    
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 64, 256, dtype=torch.float16).cuda()  
        with torch.cuda.amp.autocast():
            dummy_features = feat_extractor(dummy_input.float()) 
        FEAT_DIM = dummy_features.numel()
        del dummy_input, dummy_features 
        torch.cuda.empty_cache()
    
    print(f"특징 추출기 출력 차원 (FEAT_DIM): {FEAT_DIM}")

except Exception as e:
    print(f"Pororo 모델 로딩 실패: {e}")
    raise

# --- 2. 토크나이저 및 어휘(Vocabulary) 정의 ---
ALPH = list(" 0123456789abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ가나다라마바사아자차카타파하"
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
c2i = {c: i + 2 for i, c in enumerate(ALPH)}
PAD_TOKEN, UNK_TOKEN = 0, 1
VOCAB_SIZE = len(c2i) + 2
MAX_LEN = 30

def encode_text(text: str) -> List[int]:
    encoded = [c2i.get(char, UNK_TOKEN) for char in text]
    encoded = encoded[:MAX_LEN]
    encoded += [PAD_TOKEN] * (MAX_LEN - len(encoded))
    return encoded

# --- 3. 데이터셋 클래스 정의 (메모리 최적화) ---
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

class CustomOCRDataset(Dataset):
    def __init__(self, meta_data: List[Dict]):
        self.meta_data = meta_data

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx: int):
        item = self.meta_data[idx]
        
        img_path = item['crop_path']
        try:
            # PIL 이미지를 바로 RGB로 로드하고 즉시 변환
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                x = IMAGE_TRANSFORM(img)
        except Exception as e:
            print(f"경고: 이미지 로드 실패 - {img_path}, 오류: {e}. 더미 이미지 반환.")
            img = Image.new('RGB', (256, 64), (0, 0, 0))
            x = IMAGE_TRANSFORM(img)
            
        y = torch.tensor(encode_text(item['label']), dtype=torch.long)
        return x, y

# --- 4. 잘라낸 이미지 저장 및 메타데이터 수집 (기존과 동일) ---
def save_cropped_image(img_full_path: str, bbox_vertices: List[Dict], bbox_idx: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_full_path)
    if img is None:
        return None
    coords = np.array([[p['x'], p['y']] for p in bbox_vertices])
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    h, w, _ = img.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return None
    cropped_img = img[y0:y1, x0:x1]
    base_name = os.path.splitext(os.path.basename(img_full_path))[0]
    filename = f"{base_name}_{bbox_idx}.png"
    output_path = os.path.join(out_dir, filename)
    cv2.imwrite(output_path, cropped_img)
    return output_path

def collect_training_meta(img_root: str, json_root: str, tmp_dir: str, limit: int) -> List[Dict]:
    meta_data = []
    print(f"'{img_root}' 및 '{json_root}'에서 학습 데이터 수집 중...")
    for root, dirs, files in os.walk(json_root):
        for f_name in files:
            if not f_name.lower().endswith(('_label.json')):
                continue
            
            json_path_full = os.path.join(root, f_name)
            base_filename_no_ext = f_name.replace("_label.json", "")
            
            relative_json_dir = os.path.relpath(root, json_root)
            
            img_path_candidates = [
                os.path.join(img_root, relative_json_dir, base_filename_no_ext + ext)
                for ext in ['.png', '.jpg', '.jpeg']
            ]
            
            img_path_full = None
            for candidate_path in img_path_candidates:
                if os.path.isfile(candidate_path):
                    img_path_full = candidate_path
                    break
            
            if not img_path_full:
                continue
            
            try:
                with open(json_path_full, encoding="utf-8") as jf:
                    json_data = json.load(jf)
            except json.JSONDecodeError as e:
                print(f"경고: JSON 파일 파싱 오류 - {json_path_full}, 오류: {e}")
                continue

            for bbox_idx, item in enumerate(json_data):
                if item.get("is_corrected") and item.get("corrected_text", "").strip():
                    if 'bbox' in item and 'vertices' in item['bbox']:
                        crop_path = save_cropped_image(img_path_full, item['bbox']['vertices'], bbox_idx, tmp_dir)
                        if crop_path:
                            meta_data.append({
                                "crop_path": crop_path,
                                "label": item['corrected_text'].strip()
                            })
                            if len(meta_data) >= limit:
                                print(f"데이터 수집 한도({limit})에 도달했습니다.")
                                return meta_data
                    else:
                        print(f"경고: {json_path_full}의 bbox {bbox_idx}에 'bbox' 또는 'vertices' 키가 없습니다. 스킵합니다.")
            
    return meta_data

class CorrectionHead(nn.Module):
    def __init__(self, feature_dim: int, vocab_size: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        hidden_dim = min(feature_dim // 4, 1024) 
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_seq_len * vocab_size)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.projection(x)
        return output.view(-1, self.max_seq_len, self.vocab_size)

def train_correction_head():

    if os.path.exists(TMP_CROP):
        shutil.rmtree(TMP_CROP)
        print(f"이전 임시 크롭 폴더 '{TMP_CROP}' 삭제 완료.")

    torch.cuda.empty_cache()
    gc.collect()

    meta_data = collect_training_meta(IMG_ROOT, JSON_ROOT, TMP_CROP, SAMPLE_LIMIT)
    if not meta_data:
        print("❌ is_corrected true 인 교정된 텍스트 데이터를 찾을 수 없습니다. 학습을 종료합니다.")
        return

    random.shuffle(meta_data)
    train_size = int(0.8 * len(meta_data))
    train_meta = meta_data[:train_size]
    val_meta = meta_data[train_size:]

    if not train_meta:
        print("❌ 학습 데이터가 충분하지 않습니다. 학습을 종료합니다.")
        return
    if not val_meta:
        print("❌ 검증 데이터가 충분하지 않습니다. 학습을 계속하되 검증은 수행되지 않습니다.")
        
    print(f"✅ 총 {len(meta_data)}개의 학습 데이터를 수집했습니다.")
    print(f"✅ 학습 데이터셋 크기: {len(train_meta)}, 검증 데이터셋 크기: {len(val_meta)}")

    train_dataset = CustomOCRDataset(train_meta)
    val_dataset = CustomOCRDataset(val_meta)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH, 
        shuffle=True, 
        num_workers=0,  
        pin_memory=False, 
        drop_last=True  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False 
    )

    head_model = CorrectionHead(FEAT_DIM, VOCAB_SIZE, MAX_LEN).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    optimizer = torch.optim.AdamW(
        head_model.parameters(), 
        lr=LR,
        weight_decay=1e-3, 
        eps=1e-8 
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.cuda.amp.GradScaler(
        init_scale=2.**10, 
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )

    train_losses = []
    val_losses = []

    print("\n===== 메모리 최적화된 OCR Correction Head 학습 시작 =====")
    
    for epoch in range(1, EPOCHS + 1):

        head_model.train()
        total_train_loss = 0.0
        
        torch.cuda.empty_cache()
        gc.collect()
        
        for i, (images, labels) in enumerate(train_loader):
            try:

                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                accumulation_step = (i + 1) % ACCUMULATION_STEPS == 0

                with torch.no_grad():
                    with torch.cuda.amp.autocast():

                        if ENABLE_CHECKPOINTING:
                            features = torch.utils.checkpoint.checkpoint(lambda x: feat_extractor(x), images)
                        else:
                            features = feat_extractor(images)
                    features = features.view(images.size(0), -1)

                with torch.cuda.amp.autocast():
                    outputs = head_model(features)
                    loss = criterion(outputs.view(-1, VOCAB_SIZE), labels.view(-1))

                    loss = loss / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if accumulation_step:

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(head_model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    scheduler.step()

                total_train_loss += loss.item() * ACCUMULATION_STEPS 
                if (i + 1) % CACHE_CLEAR_FREQUENCY == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, "
                          f"Train Loss: {total_train_loss / (i+1):.4f}, LR: {current_lr:.2e}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ OOM 발생 - 배치 {i+1} 스킵. 메모리 정리 중...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        head_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                with torch.cuda.amp.autocast():
                    if ENABLE_CHECKPOINTING:
                        features = torch.utils.checkpoint.checkpoint(lambda x: feat_extractor(x), images)
                    else:
                        features = feat_extractor(images)
                    features = features.view(images.size(0), -1)
                    
                    outputs = head_model(features)
                    loss = criterion(outputs.view(-1, VOCAB_SIZE), labels.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)

        print(f"[Epoch {epoch}] 평균 Train Loss: {avg_train_loss:.4f}, 평균 Val Loss: {avg_val_loss:.4f}")
        
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            torch.save(head_model.state_dict(), f"corr_head_epoch_{epoch}.pth")
            print(f"✅ Epoch {epoch} 모델 저장 완료")

    torch.save(head_model.state_dict(), "corr_head_from_json_final.pth")
    print("✅ 최종 학습된 헤드 모델 저장 → corr_head_from_json_final.pth")
    print("\n===== OCR Correction Head 학습 완료 =====")
    
    torch.cuda.empty_cache()
    gc.collect()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()
    print("✅ Loss plot saved as 'loss_plot.png'")

def print_memory_usage():
    """GPU 메모리 사용량 출력"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        print(f"GPU 메모리 - 할당됨: {allocated:.2f}GB, 캐시됨: {cached:.2f}GB")

if __name__ == "__main__":
    print("=== 메모리 최적화 설정 적용 ===")
    print("환경 변수 및 메모리 최적화 설정 완료.")
    print_memory_usage()
    
    try:
        train_correction_head()
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise
    finally:
        print_memory_usage()