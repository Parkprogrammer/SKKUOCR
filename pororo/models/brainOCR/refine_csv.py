# import pandas as pd
# import os

# # 1. CSV 파일 로드
# csv_path = "/workspace/SKKUOCR/pororo/models/brainOCR/all_data/handwriting/labels.csv"
# df = pd.read_csv(csv_path, keep_default_na=False)

# # 2. filename 컬럼에 .png 확장자 추가
# def ensure_png(fname):
#     # 빈 값 혹은 None 처리
#     if not isinstance(fname, str) or fname.strip() == "":
#         return fname
#     # 이미 .png 로 끝나면 그대로, 아니면 .png 추가
#     return fname if fname.lower().endswith(".png") else fname + ".png"

# df["filename"] = df["filename"].apply(ensure_png)

# # 3. 변경된 데이터 저장
# out_path = os.path.splitext(csv_path)[0] + "_with_png.csv"
# df.to_csv(out_path, index=False, encoding="utf-8-sig")

# print(f"✅ 확장자 처리 완료: {out_path}")

import pandas as pd

df = pd.read_csv("/workspace/SKKUOCR/pororo/models/brainOCR/all_data/handwriting/labels.csv", nrows=5)
print(df.columns.tolist())