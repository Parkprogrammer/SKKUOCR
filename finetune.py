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
            
            # 현재 작업 디렉토리에 임시 파일 저장하도록 수정
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

    def save_correction_data_for_finetuning(self, output_dir: str):
        
        wrong_predictions = self.get_wrong_predictions()
        
        if not wrong_predictions:
            print("0.0 Error")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        img = cv2.imread(self.img_path)
        
        for i, wrong_pred in enumerate(wrong_predictions):
            
            vertices = wrong_pred['bbox']['vertices']
            points = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
            
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            margin = 5
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img.shape[1], x_max + margin)
            y_max = min(img.shape[0], y_max + margin)
            
            cropped = img[y_min:y_max, x_min:x_max]
            
            
            base_name = os.path.splitext(os.path.basename(self.img_path))[0]
            
            
            # correct_text = wrong_pred['correct_text'].replace('/', '_').replace('\\', '_')
            correct_text = wrong_pred['correct_text']
            if isinstance(correct_text, list):
                correct_text = ' '.join(str(item) for item in correct_text)
                
            safe_text = correct_text.replace('/', '_').replace('\\', '_')
            
            filename = f"{base_name}_wrong_{i}_{safe_text}.png"
            
            cv2.imwrite(os.path.join(output_dir, filename), cropped)
            
            
            metadata = {
                'original_image': self.img_path,
                'bbox': wrong_pred['bbox'],
                'wrong_prediction': wrong_pred['wrong_text'],
                'correct_text': wrong_pred['correct_text'],
                'confidence': wrong_pred['confidence']
            }
            
            metadata_file = os.path.join(output_dir, f"{base_name}_wrong_{i}_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"{len(wrong_predictions)} modified datas stroed in {output_dir}.")

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
    
    def save_all_text_data(self, output_dir: str, prefix: str = "",save_images=False):
        """
        OCR로 인식된 모든 텍스트(올바른 것과 수정된 것 모두)를 저장합니다.
        
        Args:
            output_dir: 저장할 디렉토리 경로
            prefix: 파일명 앞에 붙일 접두어 (예: 'train/image/')
            save_images: 이미지도 함께 저장할지 여부 (기본값: False)
        """
        if not self.ocr_result or not self.ocr_result.get('bounding_poly'):
            print("No OCR Results, run correction func first")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        
        if prefix:
            save_dir = os.path.join(output_dir, prefix)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = output_dir      
        
        # prefixed_base_name = f"{prefix}{base_name}" if prefix else base_name
        
        all_data = []
        
        for i, text_result in enumerate(self.ocr_result['bounding_poly']):
            
            # original_text = self.ocr_result['description'][i] if i < len(self.ocr_result['description']) else ""
            original_text = self.original_descriptions[i] if i < len(self.original_descriptions) else ""
            
            
            is_corrected = False
            corrected_text = original_text
            confidence = 1.0
            
            if i < len(self.corrections):
                corrected_text = self.corrections[i]['corrected']
                is_corrected = not self.corrections[i]['is_correct']
                confidence = self.corrections[i]['confidence']
            
            
            metadata = {
                'original_image': self.img_path,
                'bbox_index': i,
                'bbox': text_result,
                'original_text': original_text,
                'corrected_text': corrected_text,
                'is_corrected': is_corrected,
                'confidence': confidence
            }
            
            all_data.append(metadata)
            
            # NOTE: If image check is also needed
            if save_images:
                img = cv2.imread(self.img_path)
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
                
                
                safe_text = corrected_text.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                
                
                status = "corrected" if is_corrected else "original"
                filename = f"{base_name}_{i:03d}_{status}_{safe_text[:20]}.png"
                
                
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, cropped)
        
        
        all_data_filename = f"{base_name}_label.json"
        json_path = os.path.join(save_dir, all_data_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"{len(all_data)} text data saved in {json_path}.")
        
        
        txt_filename = f"{base_name}_label.txt"
        txt_path = os.path.join(save_dir, txt_filename)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Image file: {self.img_path}\n")
            f.write(f"Number of segmented texts: {len(all_data)}\n\n")
            
            for i, item in enumerate(all_data):
                f.write(f"[{i+1}] bbox {item['bbox_index']}\n")
                f.write(f"Origin text: {item['original_text']}\n")
                
                if item['is_corrected']:
                    f.write(f"Teacher Label: {item['corrected_text']} (Confidence: {item['confidence']})\n")
                else:
                    f.write("correct\n")
                
                f.write("-" * 50 + "\n")
        
        print(f"text summarization saved in {txt_path}.")
        
        return all_data
        
if __name__ == "__main__":
    
    import os
    from dotenv import load_dotenv
    
    # .env 파일 로드
    load_dotenv()
    
    GPT_API_KEY = os.getenv('OPENAI_API_KEY')
    
    ocr = PororoOcrWithCorrection(gpt_api_key=GPT_API_KEY)
    
    # IMAGE_PATH = "test/handwriting"
    BASE_IMAGE_PATH = "test"
    CORRECTION_OUTPUT_DIR = "correction_data"
    ALL_TEXT_OUTPUT_DIR = "text_data"
    
    # datasets = ["train", "test"]
    datasets = ["test"]
    categories = ["handwriting", "image", "notice"]
    
    for dataset in datasets:
        for category in categories:
            
            current_dir = os.path.join(dataset, category)
            
            
            if not os.path.exists(current_dir):
                print(f"No such Directory: {current_dir}")
                continue
                
            print(f"\n===== Processing: {current_dir} =====\n")
            
            # ex. "train/image/"
            prefix = f"{dataset}/{category}/"
            
            for filename in os.listdir(current_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(current_dir, filename)
                    print(f"Processing {filepath}...")
                    
                    result = ocr.run_ocr_corr(filepath, debug=True, use_corr=True)
                    print(f'Result for {filepath}: {result}')
                    
                    ocr.save_correction_data_for_finetuning(CORRECTION_OUTPUT_DIR)
                    
                    ocr.save_all_text_data(ALL_TEXT_OUTPUT_DIR, prefix=prefix)
                    
                    wrong_predictions = ocr.get_wrong_predictions()
                    if wrong_predictions:
                        print(f"Found Error in {len(wrong_predictions)} bbox")
                        for wp in wrong_predictions:
                            print(f"  '{wp['wrong_text']}' -> '{wp['correct_text']}'")
                    else:
                        print("0,0 error")
                    
                    print("-" * 50)