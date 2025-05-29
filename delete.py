import os

dir_path = "/workspace/SKKUOCR/correction_data/handwriting"
image_filenames = [f for f in os.listdir(dir_path) if not f.endswith(".json")]

# 삭제 대상 미리 확인
to_delete = []

for filename in os.listdir(dir_path):
    if filename.endswith("_metadata.json"):
        base = filename.replace("_metadata.json", "")
        if not any(img.startswith(base) for img in image_filenames):
            to_delete.append(filename)
    
# 삭제
for f in to_delete:
    os.remove(os.path.join(dir_path, f))
print("✅ 삭제 완료")

import os
files = os.listdir(dir_path)

# JSON과 이미지 파일 분리
json_files = [f for f in files if f.endswith("_metadata.json")]
image_files = [f for f in files if not f.endswith(".json")]

# 매칭된 쌍 수집
pairs = []

for json_file in json_files:
    base = json_file.replace("_metadata.json", "")
    for img_file in image_files:
        if img_file.startswith(base):
            pairs.append((json_file, img_file))
            break

# 리네이밍 실행
for json_file, img_file in pairs:
    base = json_file.replace("_metadata.json", "")
    
    # 확장자 추출
    img_ext = os.path.splitext(img_file)[1]
    
    # 새 이름
    new_json = os.path.join(dir_path, f"{base}.json")
    new_img = os.path.join(dir_path, f"{base}{img_ext}")
    
    # 원본 경로
    old_json = os.path.join(dir_path, json_file)
    old_img = os.path.join(dir_path, img_file)

    # 파일 존재 확인 후 리네이밍
    if os.path.exists(old_json) and os.path.exists(old_img):
        os.rename(old_json, new_json)
        os.rename(old_img, new_img)
        print(f"✅ {json_file} + {img_file} → {base}.json + {base}{img_ext}")
    else:
        print(f"❌ 파일 누락: {json_file} 또는 {img_file}")

print(f"\n총 {len(pairs)} 쌍 리네이밍 완료")