import cv2
from pororo import Pororo
from pororo.pororo import SUPPORTED_TASKS
from utils.image_util import plt_imshow, put_text
import warnings
import os

warnings.filterwarnings('ignore')


class PororoOcr:
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        self.model = model
        self.lang = lang
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)
        self.img_path = None
        self.ocr_result = {}

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=True)

        if self.ocr_result['description']:
            ocr_text = self.ocr_result["description"]
        else:
            ocr_text = "No text detected."

        if debug:
            self.show_img_with_ocr()

        return ocr_text

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path

    def show_img(self):
        plt_imshow(img=self.img_path)

    def show_img_with_ocr(self):
        img = cv2.imread(self.img_path)
        roi_img = img.copy()

        for text_result in self.ocr_result['bounding_poly']:
            text = text_result['description']
            tlX = text_result['vertices'][0]['x']
            tlY = text_result['vertices'][0]['y']
            trX = text_result['vertices'][1]['x']
            trY = text_result['vertices'][1]['y']
            brX = text_result['vertices'][2]['x']
            brY = text_result['vertices'][2]['y']
            blX = text_result['vertices'][3]['x']
            blY = text_result['vertices'][3]['y']

            pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

            topLeft = pts[0]
            topRight = pts[1]
            bottomRight = pts[2]
            bottomLeft = pts[3]

            cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)
            roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 20, font_size=15)

            # print(text)

        plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))
        
    def rotate_and_ocr(img_path):
        
        ocr_model = Pororo(task="ocr", lang="ko", model="brainocr")
        
        img = cv2.imread(img_path)
        angles = [0, 90, 180, 270]
        best_result = ""
        max_confidence = 0

        for angle in angles:
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if angle == 90 else 
                                    cv2.ROTATE_180 if angle == 180 else 
                                    cv2.ROTATE_90_COUNTERCLOCKWISE if angle == 270 else img)
            
            result = ocr_model(rotated_img)
            confidence = sum([res['score'] for res in result]) / len(result)
            
            if confidence > max_confidence:
                max_confidence = confidence
                best_result = result

        return best_result



if __name__ == "__main__":
    ocr = PororoOcr()
    # image_path = input("Enter image path: ")
    IMAGE_PATH = "test/handwriting"
    
    
    for filename in os.listdir(IMAGE_PATH):
        filepath = os.path.join(IMAGE_PATH,filename)
        filepath = "test/image/C_002.png"
        
        # if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {filepath}...")
        # result = ocr(filepath)
        # print(f"OCR result for {filepath}: {result}")
        text = ocr.run_ocr(filepath, debug=False)
        print(f'Result for {filepath} : {text}')
        #else:
            # print(f"Skipping non-image file: {filepath}")