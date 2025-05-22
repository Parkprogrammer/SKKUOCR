import gradio as gr
import cv2
import numpy as np
from pororo import Pororo
import tempfile
import os
from pyngrok import ngrok
import json
from pororo.pororo import SUPPORTED_TASKS
from utils.image_util import plt_imshow, put_text
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ngrok 설정 파일 읽기
def load_ngrok_config():
    try:
        with open('ngrok_config.json', 'r') as f:
            config = json.load(f)
            return config.get('authtoken'), config.get('domain')
    except FileNotFoundError:
        return None, None

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

        # if debug:
        #     self.show_img_with_ocr()

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

def create_interface():
    ocr = PororoOcr()
    
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 한글 OCR 텍스트 인식기
        이미지에서 텍스트를 인식하고 추출합니다.
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="입력 이미지",
                    type="numpy",
                    height=400
                )
            with gr.Column():
                image_output = gr.Image(
                    label="인식 결과",
                    type="numpy",
                    height=400
                )
        
        current_result = gr.State([])
        
        with gr.Row(equal_height=True, variant="compact"):
            with gr.Column(scale=1):
                excel_button = gr.Button("💾 엑셀 파일로 저장", variant="primary", min_width=100)
            with gr.Column(scale=2):
                download_excel = gr.File(label="", height=35)
        
        with gr.Row():
            text_output = gr.Dataframe(
                headers=["번호", "인식된 텍스트", "신뢰도"],
                label="추출 결과 미리보기",
                wrap=True
            )
            
        with gr.Row():
            json_output = gr.JSON(
                label="상세 인식 결과"
            )
            
        def process_image(image):
            if image is None:
                return (
                    None,  # image_output
                    [],    # text_output (빈 데이터프레임)
                    [],    # current_result
                    {}     # json_output (빈 JSON)
                )
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                temp_path = temp_file.name
            
            try:
                text = ocr.run_ocr(temp_path, debug=False)
                ocr_result = ocr.get_ocr_result()
                result_image = image.copy()
                
                text_results = []
                preview_data = []
                
                if 'bounding_poly' in ocr_result:
                    for idx, text_result in enumerate(ocr_result['bounding_poly'], 1):
                        vertices = text_result['vertices']
                        text = text_result['description']
                        confidence = text_result.get('score', 0)
                        
                        text_results.append({
                            'text': text,
                            'confidence': confidence,
                            'x': vertices[0]['x'],
                            'y': vertices[0]['y']
                        })
                        
                        preview_data.append([
                            idx,
                            text,
                            f"{confidence*100:.1f}%" if confidence else "N/A"
                        ])
                        
                        points = []
                        for vertex in vertices:
                            points.append([vertex['x'], vertex['y']])
                        points = np.array(points)
                        
                        cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
                        cv2.putText(result_image, f"{idx}. {text}", 
                                  (points[0][0], points[0][1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # JSON 출력을 위한 딕셔너리 생성
                json_result = {
                    "total_detected": len(text_results),
                    "texts": [result['text'] for result in text_results],
                    "details": text_results
                }
                
                return result_image, preview_data, text_results, json_result
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        def save_to_excel(text_results):
            if not text_results:
                return None
            
            df = pd.DataFrame(text_results)
            df.columns = ['텍스트', '신뢰도', 'X좌표', 'Y좌표']
            df['신뢰도'] = df['신뢰도'].apply(lambda x: f"{x*100:.1f}%" if x else "N/A")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_path = f"ocr_results_{timestamp}.xlsx"
            
            writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='OCR 결과', index=True, index_label='번호')
            
            workbook = writer.book
            worksheet = writer.sheets['OCR 결과']
            
            header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'bg_color': '#D9E1F2',
                'border': 1
            })
            
            for col_num, value in enumerate(['번호', '텍스트', '신뢰도', 'X좌표', 'Y좌표']):
                worksheet.write(0, col_num, value, header_format)
            
            for idx, col in enumerate(df.columns):
                series = df[col]
                max_len = max(
                    series.astype(str).map(len).max(),
                    len(str(series.name))
                ) + 1
                worksheet.set_column(idx, idx, max_len)
            
            writer.close()
            return excel_path
        
        image_input.change(
            fn=process_image,
            inputs=[image_input],
            outputs=[image_output, text_output, current_result, json_output]
        )
        
        excel_button.click(
            fn=save_to_excel,
            inputs=[current_result],
            outputs=[download_excel]
        )
        
        gr.HTML("""
        <div style="text-align: center; max-width: 650px; margin: 0 auto;">
            <p style="margin-bottom: 10px; font-size: 94%">
                지원되는 파일 형식: JPG | PNG | JPEG | GIF
            </p>
        </div>
        """)
    
    return demo

def setup_ngrok():
    # ngrok 설정 파일에서 인증 토큰과 도메인 읽기
    auth_token, domain = load_ngrok_config()
    
    if auth_token:
        ngrok.set_auth_token(auth_token)
    
    # 도메인이 있는 경우 해당 도메인으로 터널 생성
    if domain:
        tunnel = ngrok.connect(addr="8501", domain=domain)
    else:
        tunnel = ngrok.connect(8501)
    
    print(f' * ngrok 터널 URL: {tunnel.public_url}')
    return tunnel

if __name__ == "__main__":
    # ngrok 설정
    tunnel = setup_ngrok()
    
    try:
        # Gradio 앱 실행
        demo = create_interface()
        demo.launch(server_name="0.0.0.0", 
                   server_port=8501,
                   share=False,
                   quiet=True)
    finally:
        # 종료 시 터널 닫기
        ngrok.disconnect(tunnel.public_url)