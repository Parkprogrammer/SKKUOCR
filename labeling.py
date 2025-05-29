import os
import json
import pandas as pd

json_dir = "/workspace/SKKUOCR/correction_data/handwriting"
image_dir = "/workspace/SKKUOCR/correction_data/handwriting"

# 이미지 파일 이름들
image_filenames = os.listdir(image_dir)

new_entries = []

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = filename.replace(".json", "")  # ex: C_005_wrong_2

        # 이미지 중 정확히 같은 base로 시작하는 것이 있는지 확인
        matched = any(img.startswith(base) and not img.endswith(".json") for img in image_filenames)

        if matched:
            correct_text = data.get("correct_text", "")
            new_entries.append({"filename": base, "words": correct_text})
        else:
            print(f"⚠️ 이미지 없음: {base}")

# DataFrame 생성
new_df = pd.DataFrame(new_entries)

if not new_df.empty:
    new_df = new_df.sort_values(by="filename").reset_index(drop=True)
    new_df.to_csv("labels_handwriting.csv", index=False)
    print("✅ 존재하는 이미지에 대해서만 CSV 생성 완료: image_labels.csv")
else:
    print("❌ 유효한 매칭이 없습니다. CSV가 생성되지 않았습니다.")