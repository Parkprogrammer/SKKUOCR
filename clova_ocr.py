import argparse
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

def get_center_y(bbox):
    ys = [p["y"] for p in bbox]
    return sum(ys) / len(ys)

def group_lines_by_y(bbox_items, threshold=15):
    lines = []
    for item in sorted(bbox_items, key=lambda x: get_center_y(x["bbox"])):
        cy = get_center_y(item["bbox"])
        added = False
        for line in lines:
            avg_cy = sum(get_center_y(w["bbox"]) for w in line) / len(line)
            if abs(cy - avg_cy) < threshold:
                line.append(item)
                added = True
                break
        if not added:
            lines.append([item])
    return lines

def split_line_by_x(line, gap_threshold=40):
    sorted_line = sorted(line, key=lambda item: min(p["x"] for p in item["bbox"]))
    groups = []
    current_group = [sorted_line[0]]
    for prev, curr in zip(sorted_line[:-1], sorted_line[1:]):
        prev_x = max(p["x"] for p in prev["bbox"])
        curr_x = min(p["x"] for p in curr["bbox"])
        gap = curr_x - prev_x
        if gap > gap_threshold:
            groups.append(current_group)
            current_group = [curr]
        else:
            current_group.append(curr)
    groups.append(current_group)
    return groups

def calculate_bbox(group):
    xs = [p["x"] for item in group for p in item["bbox"]]
    ys = [p["y"] for item in group for p in item["bbox"]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, x_max, y_min, y_max 

def post_process(data):
    lines = group_lines_by_y(data, threshold=15) # group boxes with threshold_y=15 in pixel

    merged_sentences = []
    for line in lines:
        sentence_groups = split_line_by_x(line, gap_threshold=40) # split with threshold_x=40 in pixel
        for group in sentence_groups:
            merged_text = " ".join(item["text"] for item in group)
            avg_conf = sum(item["confidence"] for item in group) / len(group) # use avg confidence score 
            x_min, x_max, y_min, y_max = calculate_bbox(group)
            merged_bbox = [
                {"x": x_min, "y": y_min},
                {"x": x_max, "y": y_min},
                {"x": x_max, "y": y_max},
                {"x": x_min, "y": y_max},
            ]
            merged_sentences.append({
                "text": merged_text,
                "confidence": round(avg_conf, 4),
                "bbox": merged_bbox
            })

    return {"texts": merged_sentences, "total_texts": len(merged_sentences)}

def main(args):

    import os
    from dotenv import load_dotenv

    load_dotenv()
    file_id = os.path.splitext(args.file_name)[0]

    API_URL = os.getenv('API_URL')   
    SECRET_KEY = os.getenv('SECRET_KEY')   

    IMAGE_PATH = os.path.join(args.input_dir, args.input_type, args.file_name)
    OUTPUT_PATH = os.path.join(args.output_dir, args.input_type, f"{file_id}.json")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # IMAGE_PATH = "test/handwriting/A_005.jpg"

    print("CLOVA OCR API 호출 중...")

    result = clova_ocr_api(IMAGE_PATH, API_URL, SECRET_KEY)

    if result:
        
        extracted = extract_text_from_response(result)

        # print(extracted)
        
        if extracted:
            # print(f"총 {extracted['total_texts']}개 텍스트 발견")
            # print("\n일반 텍스트:")
            # for i, item in enumerate(extracted['texts'], 1):
            #     print(f"{i}. {item['text']} (신뢰도: {item['confidence']:.3f})")
        #    if extracted['tables']:
        #        print("\n테이블 텍스트:")
        #        for i, item in enumerate(extracted['tables'], 1):
        #            print(f"{i}. {item['text']} (행:{item['row']}, 열:{item['col']}, 신뢰도: {item['confidence']:.3f})")

            # save raw data
            with open(os.path.join(os.path.dirname(OUTPUT_PATH), f"{file_id}_raw.json"), 'w', encoding='utf-8') as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)

            # merge words into a sentence
            merged_output = post_process(extracted["texts"])
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(merged_output, f, ensure_ascii=False, indent=2)
            
            print(f"\n결과가 {OUTPUT_PATH}에 저장되었습니다.")
        
        else:
            print("텍스트 추출 실패")

    else:
        print("API 호출 실패")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clova OCR API")
    parser.add_argument("--input_dir", type=str, default="test")
    parser.add_argument("--input_type", type=str, default="handwriting")
    parser.add_argument("--file_name", type=str, default="A_005.jpg")
    parser.add_argument("--output_dir", type=str, default="test_CLOVA")
    
    args = parser.parse_args() 
    main(args)