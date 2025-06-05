import pandas as pd
from pathlib import Path

directory = Path('/workspace/SKKUOCR2/SKKUOCR_fine/CLOVA_V3_train/merged_images')

df = pd.read_csv('/workspace/SKKUOCR2/SKKUOCR_fine/CLOVA_V3_train/train_labels.csv')

def file_exists(filename: str) -> bool:
    return (directory / filename).exists()

filtered_df = df[df['filename'].apply(file_exists)]

print(f"원본 행 개수: {len(df)}, 필터링 후 행 개수: {len(filtered_df)}")
filtered_df.to_csv('filtered_labels_filtered.csv', index=False)