import requests
import json
import base64
import time
import uuid

def clova_ocr_api(image_path, api_url, secret_key):
    """
    CLOVA OCR API를 사용한 텍스트 추출
    """

    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()

    request_json = {
        'version': 'V2',
        'requestId': str(uuid.uuid4()),
        'timestamp': int(time.time() * 1000),
        'lang': 'ko',
        'images': [
            {
                'format': 'png',
                'name': 'demo',
                'data': img_data
            }
        ],
        'enableTableDetection': False
    }

    headers = {
        'X-OCR-SECRET': secret_key,
        'Content-Type': 'application/json'
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(request_json))

    if response.status_code == 200:
        return response.json()
    else:
        print(f"API 호출 실패: {response.status_code}")
        print(f"응답: {response.text}")
        return None

def extract_text_from_response(response_data):
        """
        API 응답에서 텍스트 추출
        """

        if not response_data or 'images' not in response_data:
            return None

        image_result = response_data['images'][0]

        if image_result['inferResult'] != 'SUCCESS':
            print(f"OCR 실패: {image_result['message']}")
            return None

        extracted_texts = []

        if 'fields' in image_result:
            for field in image_result['fields']:
                if field['inferConfidence'] > 0.5:  # 50% >
                    extracted_texts.append({
                        'text': field['inferText'],
                        'confidence': field['inferConfidence'],
                        'bbox': field.get('boundingPoly', {}).get('vertices', [])
                    })

        # 테이블 정보 추출
        #    table_data = []
        #    if 'tables' in image_result:
        #        for table in image_result['tables']:
        #            for cell in table.get('cells', []):
        #                for line in cell.get('cellTextLines', []):
        #                    for word in line.get('cellWords', []):
        #                        if word['inferConfidence'] > 0.5:
        #                            table_data.append({
        #                                'text': word['inferText'],
        #                                'confidence': word['inferConfidence'],
        #                                'row': cell.get('rowIndex', -1),
        #                                'col': cell.get('columnIndex', -1)
        #                            })

        return {
            'texts': extracted_texts,
        #    'tables': table_data,
            'total_texts': len(extracted_texts) # + len(table_data)
        }

def main():

    import os
    from dotenv import load_dotenv

    load_dotenv()

    API_URL = os.getenv('API_URL')   
    SECRET_KEY = os.getenv('SECRET_KEY')   

    IMAGE_PATH = "test/handwriting/A_005.jpg"

    print("CLOVA OCR API 호출 중...")

    result = clova_ocr_api(IMAGE_PATH, API_URL, SECRET_KEY)

    if result:
        
        extracted = extract_text_from_response(result)

        print(extracted)
        
        if extracted:
            print(f"총 {extracted['total_texts']}개 텍스트 발견")
            
            
            print("\n일반 텍스트:")
            for i, item in enumerate(extracted['texts'], 1):
                print(f"{i}. {item['text']} (신뢰도: {item['confidence']:.3f})")
            
        #    if extracted['tables']:
        #        print("\n테이블 텍스트:")
        #        for i, item in enumerate(extracted['tables'], 1):
        #            print(f"{i}. {item['text']} (행:{item['row']}, 열:{item['col']}, 신뢰도: {item['confidence']:.3f})")
            
            with open('clova_ocr_result.json', 'w', encoding='utf-8') as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)
            
            print("\n결과가 clova_ocr_result.json에 저장되었습니다.")
        
        else:
            print("텍스트 추출 실패")

    else:
        print("API 호출 실패")

if __name__ == "__main__":
    main()