# Efficient OCR for Information Digitalization

SKKU OCR 프로젝트는 제한된 자원 환경에서 다국어 텍스트 인식의 성능을 향상시키기 위한 연구입니다. 특히 한국어와 같은 복잡한 언어에서 발생하는 클래스 불균형과 특수문자 인식 문제를 해결하기 위해 혁신적인 Greedy Model Selection Algorithm을 제안합니다.

## 🎯 주요 기여

### 1. Greedy Model Selection Algorithm
- **3단계 계층적 예측**: 특수문자 → 숫자 → 기본 모델 순서로 실행
- **신뢰도 기반 모델 선택**: 각 단계별 confidence threshold를 통한 최적 결과 선택
- **10배 향상된 효율성**: 기존 방법 대비 계산 비용 대폭 절감

### 2. 다국어 클래스 불균형 해결
- **통계적 분석**: 영어(36개), 한국어(2393개) 문자 클래스 분석
- **Confidence Distribution 분석**: 텍스트, 숫자, 특수문자별 신뢰도 패턴 발견
- **T-test 검증**: Symbol Present vs Text Dominant 카테고리 간 유의미한 차이 확인 (p=0.0188)

### 3. 효율적 Fine-tuning 전략
- **GPT-4 기반 자동 레이블링**: 잘못된 OCR 결과 자동 수정
- **노이즈 필터링**: Confidence threshold (≈1/2900) 기반 OOV 토큰 제거
- **하이퍼파라미터 최적화**: Grid search를 통한 최적 학습률(1e-6), 배치 크기(32), 에포크(200) 도출

## 📊 실험 결과

### 성능 향상
- **2-path 모델**: 기존 대비 1% 정확도 향상
- **3-path 모델**: 추가 1% 향상으로 총 2% 성능 개선
- **특수문자 인식**: '/'와 '1' 구분 등 기존 오류 해결

### 신뢰도 분석
- **Text Dominant**: 높은 신뢰도와 낮은 분산
- **Symbol Present**: 낮은 신뢰도와 높은 분산
- **Cohen's d**: 0.2055 (medium effect size)

## 🏗️ 시스템 아키텍처

### Detection Module
- **CRAFT 기반**: Region score와 Affinity score를 통한 텍스트 영역 검출
- **U-Net 구조**: VGG16 백본과 업스케일링 컨볼루션

### Recognition Module


## 📦 설치 및 사용법

### 요구사항
```bash
pip install torch torchvision opencv-python pillow
pip install numpy pandas scikit-image wandb
pip install openai python-dotenv  # GPT-4 기반 레이블링용
```
환경 설정
bash# .env 파일 생성
```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```
데이터셋 준비
bash# GPT-4 기반 OCR 결과 수정 및 데이터 수집
```bash
python finetune.py
```
모델 학습
bash# 기본 파인튜닝
```bash
python finetune_brainocr.py \
    --train_root CLOVA_V3_train \
    --test_root CLOVA_V2_test \
    --epochs 200 \
    --batch 32 \
    --lr 1e-6
```
SKKUOCR/
├── pororo/                           # 메인 OCR 라이브러리
│   ├── models/brainOCR/             # BrainOCR 모델 구현
│   │   ├── brainocr.py             # Reader 클래스
│   │   ├── recognition.py          # 인식 모델 및 예측 함수
│   │   ├── detection.py            # CRAFT 검출 모델
│   │   ├── model.py                # 모델 아키텍처
│   │   └── modules/                # 서브모듈들
│   └── tasks/                       # 태스크별 팩토리 클래스
│       └── optical_character_recognition.py
├── datasets.py                      # 데이터셋 및 전처리 유틸리티
├── finetune_brainocr.py            # 모델 파인튜닝 스크립트
├── finetune.py                     # GPT-4 기반 데이터 수집
├── main.py                         # 기본 OCR 실행 스크립트
└── utils/                          # 유틸리티 함수들
    └── image_util.py              # 이미지 처리 유틸리티
