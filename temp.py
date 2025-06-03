import re

def analyze_prediction_results(file_path):
    """
    예측 결과를 파싱하여 Confidence Score, Accuracy 등을 계산합니다.

    Args:
        file_path (str): 분석할 텍스트 파일의 경로.

    Returns:
        dict: 분석 결과가 담긴 딕셔너리.
    """
    correct_confidences = []
    incorrect_confidences = []
    total_confidences = []
    total_predictions = 0
    correct_predictions = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 각 항목을 정규 표현식으로 추출
    # GT: (.*?)
    # PR: (.*?)
    # CONF:([\d.]+)
    # 여기서 (.*?)는 최소 일치, [\d.]+는 숫자 또는 점에 일치
    # 각 항목은 '----------------------------------------'로 구분되므로 이를 활용
    entries = re.findall(r"GT: (.*?)\nPR: (.*?)\nCONF:([\d.]+)\s*.\n", content, re.DOTALL)

    for gt, pr, conf_str in entries:
        conf = float(conf_str)
        total_confidences.append(conf)
        total_predictions += 1

        if gt == pr:
            correct_predictions += 1
            correct_confidences.append(conf)
        else:
            incorrect_confidences.append(conf)

    # 결과 계산
    avg_correct_confidence = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    avg_incorrect_confidence = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions else 0
    overall_avg_confidence = sum(total_confidences) / len(total_confidences) if total_confidences else 0

    results = {
        "평균_일치_Confidence": avg_correct_confidence,
        "평균_불일치_Confidence": avg_incorrect_confidence,
        "Accuracy": accuracy,
        "전체_평균_Confidence": overall_avg_confidence,
        "총_예측_수": total_predictions,
        "정답_예측_수": correct_predictions,
        "오답_예측_수": total_predictions - correct_predictions
    }
    return results

# 예시 파일 생성 (실제 파일 경로에 맞게 수정)
file_content = """✓ Saved ckpt → assets/finetuned_brainocr.pt
✓ Saved opt  → assets/finetuned_opt.txt
GT: Selection
PR: Selection
CONF:0.468  ✓
----------------------------------------
GT: 2.
PR: 2'
CONF:0.570  ✗
----------------------------------------
GT: Worst-Case
PR: @orst - Case
CONF:0.096  ✗
----------------------------------------
GT: Example
PR: Example
CONF:0.800  ✓
----------------------------------------
GT: Test
PR: Tost
CONF:0.300  ✗
"""

with open("hi.txt", "w", encoding="utf-8") as f:
    f.write(file_content)

# 함수 호출 및 결과 출력
file_path = "text1.txt"
analysis_results = analyze_prediction_results(file_path)

print(f"평균 일치 Confidence Score: {analysis_results['평균_일치_Confidence']:.3f}")
print(f"평균 불일치 Confidence Score: {analysis_results['평균_불일치_Confidence']:.3f}")
print(f"Accuracy: {analysis_results['Accuracy']:.2f}%")
print(f"전체 평균 Confidence Score: {analysis_results['전체_평균_Confidence']:.3f}")
print(f"총 예측 수: {analysis_results['총_예측_수']}")
print(f"정답 예측 수: {analysis_results['정답_예측_수']}")
print(f"오답 예측 수: {analysis_results['오답_예측_수']}")