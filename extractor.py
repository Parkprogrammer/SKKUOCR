import cv2
import base64
import json
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from pororo import Pororo
from pororo.pororo import SUPPORTED_TASKS
from utils.image_util import plt_imshow, put_text
import warnings
import numpy as np


warnings.filterwarnings('ignore')

class GPTOCRCorrector:
    
    #TODO: Change model from gpt-4-preview to ???
    # def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
      
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def encode_img_base64(self, image_pth: str) -> str:
        
        with open(image_pth, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def crop_txt_region(self, image_pth: str,bbox: List[Dict]) -> List[np.ndarray]:
        
        img = cv2.imread(image_pth)
        cropped_images = []
        
        for text_result in bbox:
            vertices = text_result['vertices']
            points = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
            
            # Calculate Bounding box
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # Add margin
            margin = 5
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img.shape[1], x_max + margin)
            y_max = min(img.shape[0], y_max + margin)
            
            cropped = img[y_min:y_max, x_min:x_max]
            cropped_images.append(cropped)
            
        return cropped_images
    
    # Temporarily store bbox to send to Foundation models for fune-tuning.
    def save_cropped_images(self, cropped_images: List[np.ndarray], base_path: str) -> List[str]:
        
        temp_dir = "temp_crops"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_paths = []
        base_name = os.path.basename(base_path).replace('.', '_')
        
        for i, cropped in enumerate(cropped_images):
            temp_path = os.path.join(temp_dir, f"{base_name}_crop_{i}.png")
            try:
                cv2.imwrite(temp_path, cropped)
                temp_paths.append(temp_path)
            except Exception as e:
                print(f"Temp saving cropped image failed: {temp_path}, 에러: {e}")
                
        return temp_paths
    
    def verify_ocr_with_gpt(self, cropped_image_paths: List[str], ocr_texts: List[str]) -> List[Dict]:
        """
        GPT API를 사용하여 OCR 결과 검증
        
        Args:
            cropped_image_paths: 크롭된 이미지 파일 경로들
            ocr_texts: OCR로 인식된 텍스트들
            
        Returns:
            List[Dict]: [{"original": "OCR결과", "corrected": "수정된결과", "is_correct": bool}, ...]
        """
        corrections = []
        
        for img_path, ocr_text in zip(cropped_image_paths, ocr_texts):
            base64_image = self.encode_img_base64(img_path)
            
            prompt = f"""
                이 이미지에 있는 텍스트를 정확히 읽어주세요.
                OCR 시스템이 이 텍스트를 "{ocr_text}"로 인식했습니다.

                다음 형식으로 JSON 응답해주세요:
                {{
                    "actual_text": "이미지에서 실제로 보이는 정확한 텍스트",
                    "is_ocr_correct": true/false,
                    "confidence": 0.0-1.0
                }}

                한국어와 영어, 숫자 및 기호를 모두 정확히 인식해주세요.
                """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                response_text = response.choices[0].message.content
                
                # markdown 코드 블록 제거
                clean_text = response_text.strip()
                if clean_text.startswith('```'):
                    lines = clean_text.split('\n')
                    clean_text = '\n'.join(lines[1:-1])
                
                try:
                    result = json.loads(clean_text)
                    corrections.append({
                        "original": ocr_text,
                        "corrected": result["actual_text"],
                        "is_correct": result["is_ocr_correct"],
                        "confidence": result.get("confidence", 0.5)
                    })
                except json.JSONDecodeError:
                    corrections.append({
                        "original": ocr_text,
                        "corrected": ocr_text,
                        "is_correct": True,
                        "confidence": 0.5
                    })
                    
            except Exception as e:
                print(f"GPT API 호출 실패: {e}")
                corrections.append({
                    "original": ocr_text,
                    "corrected": ocr_text,
                    "is_correct": True,
                    "confidence": 0.0
                })
        
        return corrections

class PororoOcrWithCorrection:
    
    def __init__(self, model: str = "brainocr", lang: str = "ko", gpt_api_key: str = None, **kwargs):
        self.model = model
        self.lang = lang
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)
        self.img_path = None
        self.ocr_result = {}
        self.corrections = []
                
        if gpt_api_key:
            self.gpt_corrector = GPTOCRCorrector(gpt_api_key)
        else:
            self.gpt_corrector = None
            print("Warning: GPT API key has not been set, no correction")
            
    def run_ocr_corr(self, img_path: str, debug: bool = False, use_corr: bool = True):
        
        self.img_path = img_path
        
        self.ocr_result = self._ocr(img_path, detail=True)
        
        if not self.ocr_result['description']:
            return "No text detected."
        
        self.original_descriptions = self.ocr_result['description'].copy()
        
        if use_corr and self.gpt_corrector:
            cropped_images = self.gpt_corrector.crop_txt_region(
                img_path, self.ocr_result['bounding_poly']
            )
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            temp_paths = self.gpt_corrector.save_cropped_images(
                cropped_images, base_name
            )
            
            try:
                self.corrections = self.gpt_corrector.verify_ocr_with_gpt(
                    temp_paths, self.ocr_result['description']
                )
                
                corrected_texts = []
                for i, correction in enumerate(self.corrections):
                    if not correction['is_correct']:
                        corrected_texts.append(correction['corrected'])
                        self.ocr_result['bounding_poly'][i]['description'] = correction['corrected']
                        print(f"Modified: '{correction['original']}' -> '{correction['corrected']}'")
                    else:
                        corrected_texts.append(correction['original'])
                
                self.ocr_result['description'] = corrected_texts
                
            finally:
                for temp_path in temp_paths:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        if debug:
            self.show_img_with_ocr_and_corrections()
        
        return self.ocr_result['description']
    
    def get_wrong_predictions(self) -> List[Dict]:
        
        if not self.corrections:
            return []
        
        wrong_predictions = []
        for i, correction in enumerate(self.corrections):
            if not correction['is_correct']:
                wrong_predictions.append({
                    'bbox': self.ocr_result['bounding_poly'][i],
                    'wrong_text': correction['original'],
                    'correct_text': correction['corrected'],
                    'confidence': correction['confidence']
                })
        
        return wrong_predictions

    def save_combined_data(self, output_base_dir: str, dataset: str, category: str, image_counter: List[int]):
        """
        모든 bbox 이미지와 메타데이터를 A_XXX_YYY.png 형식으로 저장
        
        Args:
            output_base_dir: 기본 저장 디렉토리 경로 (correction_data_a)
            dataset: 데이터셋 이름 (test, train)
            category: 카테고리 이름 (handwriting, image, notice)
            image_counter: 전체 이미지 카운터 (리스트로 전달하여 참조 유지)
        """
        if not self.ocr_result or not self.ocr_result.get('bounding_poly'):
            print("No OCR Results, run correction func first")
            return
        
        image_output_dir = os.path.join(output_base_dir, dataset, category)
        os.makedirs(image_output_dir, exist_ok=True)
        
        label_output_dir = os.path.join(output_base_dir, dataset, category, "label")
        os.makedirs(label_output_dir, exist_ok=True)
        
        img = cv2.imread(self.img_path)
        all_data = []
        
        for i, text_result in enumerate(self.ocr_result['bounding_poly']):

            vertices = text_result['vertices']
            points = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
            
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            margin = 5
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img.shape[1], x_max + margin)
            y_max = min(img.shape[0], y_max + margin)
            
            cropped = img[y_min:y_max, x_min:x_max]
            
            filename = f"A_{image_counter[0]:03d}_{i:03d}.png"
            output_path = os.path.join(image_output_dir, filename)
            cv2.imwrite(output_path, cropped)
            
            original_text = self.original_descriptions[i] if i < len(self.original_descriptions) else ""
            
            is_corrected = False
            corrected_text = original_text
            confidence = 1.0
            
            if i < len(self.corrections):
                corrected_text = self.corrections[i]['corrected']
                is_corrected = not self.corrections[i]['is_correct']
                confidence = self.corrections[i]['confidence']
            
            metadata = {
                'image_filename': filename,
                'original_image': self.img_path,
                'bbox_index': i,
                'bbox': text_result,
                'original_text': original_text,
                'corrected_text': corrected_text,
                'is_corrected': is_corrected,
                'confidence': confidence
            }
            
            all_data.append(metadata)
        
        # 메타데이터 JSON 저장 (label 폴더에)
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        json_filename = f"A_{image_counter[0]:03d}_metadata.json"
        json_path = os.path.join(label_output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(all_data)} bbox images to {image_output_dir}")
        print(f"Saved metadata to {json_path}")
        
        image_counter[0] += 1
        
        return all_data

    def show_img_with_ocr_and_corrections(self):
        
        img = cv2.imread(self.img_path)
        roi_img = img.copy()

        for i, text_result in enumerate(self.ocr_result['bounding_poly']):
            text = text_result['description']
            vertices = text_result['vertices']
            
            tlX, tlY = vertices[0]['x'], vertices[0]['y']
            trX, trY = vertices[1]['x'], vertices[1]['y']
            brX, brY = vertices[2]['x'], vertices[2]['y']
            blX, blY = vertices[3]['x'], vertices[3]['y']

            pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

            
            if i < len(self.corrections) and not self.corrections[i]['is_correct']:
                color = (0, 0, 255)
                text = f"{text} (modified)"
            else:
                color = (0, 255, 0)  

            cv2.line(roi_img, pts[0], pts[1], color, 2)
            cv2.line(roi_img, pts[1], pts[2], color, 2)
            cv2.line(roi_img, pts[2], pts[3], color, 2)
            cv2.line(roi_img, pts[3], pts[0], color, 2)
            
            roi_img = put_text(roi_img, text, pts[0][0], pts[0][1] - 20, font_size=15)

        # plt_imshow(["Original", "OCR with Corrections"], [img, roi_img], figsize=(16, 10))

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()
        
if __name__ == "__main__":
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    GPT_API_KEY = os.getenv('OPENAI_API_KEY')
    
    ocr = PororoOcrWithCorrection(gpt_api_key=GPT_API_KEY)
    
    BASE_IMAGE_PATH = "train"
    OUTPUT_DIR = "correction_data_a"
    
    datasets = ["train"]
    categories = ["handwriting", "image", "notice"]
    
    image_counter = [0]
    
    for dataset in datasets:
        for category in categories:
            
            current_dir = os.path.join(dataset, category)
            
            if not os.path.exists(current_dir):
                print(f"No such Directory: {current_dir}")
                continue
                
            print(f"\n===== Processing: {current_dir} =====\n")
            
            for filename in os.listdir(current_dir):

                if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.upper().startswith('A'):
                    filepath = os.path.join(current_dir, filename)
                    print(f"Processing {filepath}...")
                    
                    result = ocr.run_ocr_corr(filepath, debug=True, use_corr=True)
                    print(f'Result for {filepath}: {result}')
                    
                    ocr.save_combined_data(OUTPUT_DIR, dataset, category, image_counter)
                    
                    wrong_predictions = ocr.get_wrong_predictions()
                    if wrong_predictions:
                        print(f"Found Error in {len(wrong_predictions)} bbox")
                        for wp in wrong_predictions:
                            print(f"  '{wp['wrong_text']}' -> '{wp['correct_text']}'")
                    else:
                        print("0,0 error")
                    
                    print("-" * 50)
                else:
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    print(f"Skipping {filename} (not starting with 'A')")
    
    print(f"\nTotal processed A-prefixed images: {image_counter[0]}")