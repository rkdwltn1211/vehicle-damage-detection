# 🚗 CarGuard — Vehicle Damage Detection System

> **탁송 전·후 차량 이미지를 AI로 분석하여 흠집 심각도(0~4단계)를 자동 판정하는 웹 서비스**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![AWS](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat-square&logo=amazonaws&logoColor=white)

**🌐 Live Demo**: [http://3.27.11.142:8000](http://3.27.11.142:8000)

---

## 📌 프로젝트 개요

탁송 서비스에서 차량 인수·반납 시 발생하는 **손상 책임 분쟁**을 해결하기 위해 개발한 AI 기반 차량 흠집 탐지 시스템입니다.

탁송 전·후 차량 이미지를 업로드하면 **EfficientNet-B0 CNN 모델**이 손상 여부와 심각도를 0~4단계로 자동 판정하고, 결과를 시각화하여 반환합니다.

> 🗓 **개발 기간**: 2026.01 – 2026.04  
> 👤 **개발자**: 강지수 (개인 프로젝트 — 팀 프로젝트에서 담당한 파트를 독립 서비스로 고도화)

---

## 🏗 시스템 아키텍처

```
사용자 (브라우저)
      │
      │  이미지 업로드 (before / after)
      ▼
FastAPI 백엔드 (main.py)
      │
      ├─── EfficientNet-B0 CNN ──► 심각도 판정 (0~4)
      │
      └─── OpenCV 파이프라인 ──► 흠집 위치 시각화
      │
      ▼
결과 반환 (JSON + 비교 이미지 base64)
      │
      ▼
프론트엔드 시각화 (index.html)
```

---

## 🧠 핵심 기술

### ① CNN 모델 — 심각도 분류 (EfficientNet-B0)

- ImageNet pretrained **EfficientNet-B0** Fine-tuning (Transfer Learning)
- 클래스 불균형 대응: `CrossEntropyLoss` 클래스 가중치 적용
- Early Stopping (patience=5) 적용
- **val accuracy: 64%** (126쌍 소규모 데이터 기준)

### ② 데이터 파이프라인 (OpenCV)

```
before 이미지 + after 이미지
        │
        ▼
① ECC 정렬          촬영 각도 차이 자동 보정
        │
        ▼
② Canny 엣지 탐지   흠집 영역 마스크 생성
        │
        ▼
③ Morphology 필터   노이즈 제거
        │
        ▼
④ ROI 패치 추출     흠집 영역 크롭
        │
        ▼
⑤ 데이터 증강       634개 → 4,190장
```

### ③ 심각도 라벨링 체계 (직접 설계)

| 단계 | 수준 | 기준 |
|:---:|------|------|
| **0** | 정상 | 신규 흠집 없음 |
| **1** | 경미 | 얕은 스크래치, 멀리서 잘 안 보임 |
| **2** | 보통 | 육안으로 명확, 도장 손상, 부분 도색 가능 |
| **3** | 심각 | 넓거나 깊음, 찌그러짐, 수리 필요 |
| **4** | 매우 심각 | 구조적 손상 수준, 즉시 수리·교체 필요 |

---

## 📊 데이터셋

| 항목 | 내용 |
|------|------|
| 원본 데이터 | 차량 126쌍 × 6방향 = 756쌍 |
| 라벨링 | 심각도 0~4단계 직접 설계 |
| 증강 후 | 4,190장 (train 2,900 / val 380) |
| 증강 방법 | 좌우반전, 상하반전, 회전(±15°), 밝기조절, 가우시안블러 |

| 심각도 | Score 0 | Score 1 | Score 2 | Score 3 | Score 4 |
|:------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 대수 | 46대 | 22대 | 23대 | 21대 | 14대 |

---

## 🌐 API 명세

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/analyze` | before/after 이미지 업로드 → 심각도 분석 결과 반환 |
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/` | 프론트엔드 서빙 |

**POST /analyze 응답 예시**
```json
{
  "before": { "severity": 0, "label": "Normal", "confidence": 1.0 },
  "after": { "severity": 4, "label": "Critical", "confidence": 0.91 },
  "damage_detected": true,
  "severity_change": 4,
  "result_image": "<base64 encoded image>"
}
```

---

## 🛠 Tech Stack

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.12 |
| 딥러닝 | PyTorch, EfficientNet-B0 (torchvision) |
| 이미지 처리 | OpenCV, PIL |
| 백엔드 | FastAPI, Uvicorn |
| 프론트엔드 | HTML / CSS / JavaScript |
| 클라우드 | AWS EC2 (t3.micro, Ubuntu 22.04) |
| 데이터 분석 | Pandas, NumPy |

---

## 📂 프로젝트 구조

```
vehicle-damage-detection/
├── main.py                  # FastAPI 서버 (API 엔드포인트)
├── index.html               # 프론트엔드
├── labels.csv               # 심각도 라벨 (car_id, score)
├── step1_prepare_data.py    # 데이터 준비 (OpenCV 파이프라인 + 증강)
├── step2_train_cnn.py       # CNN 모델 학습 (EfficientNet-B0)
├── step3_pipeline.py        # 추론 파이프라인
├── model/
│   └── best_model.pth       # 학습된 CNN 모델
└── README.md
```

---

## 🚀 로컬 실행 방법

```bash
# 1. 레포 클론
git clone https://github.com/rkdwltn1211/vehicle-damage-detection.git
cd vehicle-damage-detection

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install fastapi uvicorn torch torchvision opencv-python-headless pillow numpy python-multipart

# 4. 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속

---

## ☁️ 배포 환경 (AWS EC2)

| 항목 | 내용 |
|------|------|
| 서버 | AWS EC2 t3.micro |
| OS | Ubuntu Server 22.04 LTS |
| 프로세스 관리 | systemd (서버 재시작 시 자동 실행) |
| 포트 | 8000 |

---

## ⚠️ 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|----------|
| val accuracy 64% — 126쌍 소규모 데이터 한계 | CarDD 등 공개 데이터셋 추가 결합 |
| Score 3·4 F1 낮음 — 심각 케이스 데이터 부족 | 데이터 추가 수집 및 Focal Loss 적용 |
| CNN과 OpenCV 파이프라인 미통합 | 단일 파이프라인으로 통합 예정 |

---

## 🔗 관련 레포지토리

이 프로젝트는 아래 팀 프로젝트에서 담당한 파트를 독립 서비스로 고도화한 것입니다.

👉 [vehicle-delivery-ai-system](https://github.com/rkdwltn1211/vehicle-delivery-ai-system) — AI 기반 탁송 매칭 플랫폼 (4인 팀 프로젝트)

---

## 👤 개발자

**강지수 (Kang Ji Soo)**  
📧 rkdwl3264@naver.com  
🐙 [github.com/rkdwltn1211](https://github.com/rkdwltn1211)
