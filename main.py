"""
FastAPI 백엔드 — 차량 흠집 비교 분석
=====================================
before/after 이미지를 업로드하면
CNN으로 심각도를 판정하고 결과를 반환합니다.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os

# =====================================================================
# ✅ 경로 설정
# =====================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")

# =====================================================================
# 설정
# =====================================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
IMG_SIZE    = 224
CHANGE_THRESHOLD = 2
CONFIDENCE_THRESHOLD = 0.8

SEVERITY_INFO = {
    0: {"label": "Normal",   "color": "#22c55e", "desc": "No damage detected."},
    1: {"label": "Minor",    "color": "#facc15", "desc": "Minor scratch. Polish may fix it."},
    2: {"label": "Moderate", "color": "#f97316", "desc": "Visible damage. Partial repaint needed."},
    3: {"label": "Severe",   "color": "#ef4444", "desc": "Severe damage. Professional repair needed."},
    4: {"label": "Critical", "color": "#991b1b", "desc": "Critical damage. Immediate repair required."},
}

# =====================================================================
# FastAPI 앱
# =====================================================================
app = FastAPI(title="Car Scratch Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# CNN 모델 로드
# =====================================================================
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"✅ 모델 로드 완료 (val_acc: {checkpoint.get('val_acc', 0):.3f})")
    return model

cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

model = load_model()

def predict(pil_img):
    tensor = cnn_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    return int(pred.item()), float(conf.item()), probs.cpu().numpy()[0].tolist()

def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def make_result_image(pil_before, pil_after, before_sev, after_sev, before_conf, after_conf):
    """결과 비교 이미지 생성 → base64 반환"""
    PANEL_W, PANEL_H = 820, 420
    IMG_W,   IMG_H   = 360, 270
    MARGIN           = 20

    panel = np.ones((PANEL_H, PANEL_W, 3), dtype=np.uint8) * 25

    # Before 이미지
    b_cv = cv2.cvtColor(np.array(pil_before.resize((IMG_W, IMG_H))), cv2.COLOR_RGB2BGR)
    panel[50:50+IMG_H, MARGIN:MARGIN+IMG_W] = b_cv

    # After 이미지
    a_cv = cv2.cvtColor(np.array(pil_after.resize((IMG_W, IMG_H))), cv2.COLOR_RGB2BGR)
    ax = PANEL_W - MARGIN - IMG_W
    panel[50:50+IMG_H, ax:ax+IMG_W] = a_cv

    # 화살표
    mid_x, mid_y = PANEL_W // 2, 50 + IMG_H // 2
    cv2.arrowedLine(panel, (mid_x-28, mid_y), (mid_x+28, mid_y), (130,130,130), 2, tipLength=0.4)

    # Before 라벨
    b_info = SEVERITY_INFO[before_sev]
    b_color_bgr = hex_to_bgr(b_info["color"])
    cv2.rectangle(panel, (MARGIN, 50+IMG_H), (MARGIN+IMG_W, 50+IMG_H+28), (45,45,45), -1)
    cv2.putText(panel, f"BEFORE: {b_info['label']} (Lv.{before_sev} | {before_conf*100:.0f}%)",
                (MARGIN+6, 50+IMG_H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color_bgr, 1)

    # After 라벨
    a_info = SEVERITY_INFO[after_sev]
    a_color_bgr = hex_to_bgr(a_info["color"])
    cv2.rectangle(panel, (ax, 50+IMG_H), (ax+IMG_W, 50+IMG_H+28), (45,45,45), -1)
    cv2.putText(panel, f"AFTER:  {a_info['label']} (Lv.{after_sev} | {after_conf*100:.0f}%)",
                (ax+6, 50+IMG_H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, a_color_bgr, 1)

    # 변화 텍스트
    diff = after_sev - before_sev
    change_y = 50 + IMG_H + 65
    if after_sev >= CHANGE_THRESHOLD and diff > 0 and after_conf >= CONFIDENCE_THRESHOLD:
        change_text  = f"Severity INCREASED  +{diff} level"
        change_color = (80, 80, 240)
    else:
        change_text  = "No Change Detected"
        change_color = (160, 160, 160)

    cv2.rectangle(panel, (MARGIN, change_y-26), (PANEL_W-MARGIN, change_y+8), (45,45,45), -1)
    cv2.putText(panel, change_text, (PANEL_W//2-170, change_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, change_color, 2)

    # 심각도 바
    bar_y  = change_y + 40
    bar_w  = (PANEL_W - MARGIN*2) // 5
    colors_bgr = [(94,197,34),(21,204,250),(22,115,249),(68,68,239),(27,27,153)]
    for i in range(5):
        bx1 = MARGIN + i * bar_w
        bx2 = bx1 + bar_w - 4
        col = colors_bgr[i] if i == after_sev else (65,65,65)
        cv2.rectangle(panel, (bx1, bar_y), (bx2, bar_y+24), col, -1)
        if i == after_sev:
            cv2.rectangle(panel, (bx1-2, bar_y-2), (bx2+2, bar_y+26), colors_bgr[i], 2)
        cv2.putText(panel, str(i), (bx1+bar_w//2-5, bar_y+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255,255,255) if i == after_sev else (100,100,100), 1)

    # base64 변환
    _, buf = cv2.imencode(".jpg", panel)
    return base64.b64encode(buf).decode("utf-8")

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)

# =====================================================================
# API 엔드포인트
# =====================================================================
@app.post("/analyze")
async def analyze(
    before: UploadFile = File(...),
    after:  UploadFile = File(...),
):
    # 파일 타입 검증
    for f in [before, after]:
        if not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    # 이미지 로드
    before_pil = Image.open(io.BytesIO(await before.read())).convert("RGB")
    after_pil  = Image.open(io.BytesIO(await after.read())).convert("RGB")

    # CNN 판정
    before_sev, before_conf, before_probs = predict(before_pil)
    after_sev,  after_conf,  after_probs  = predict(after_pil)

    # before는 정상(Lv.0) 고정
    before_sev, before_conf = 0, 1.0

    # 변화 감지
    diff = after_sev - before_sev
    damage_detected = (
        after_sev >= CHANGE_THRESHOLD and
        diff > 0 and
        after_conf >= CONFIDENCE_THRESHOLD
    )

    # 결과 이미지 생성
    result_image_b64 = make_result_image(
        before_pil, after_pil,
        before_sev, after_sev,
        before_conf, after_conf
    )

    return {
        "before": {
            "severity": before_sev,
            "label":    SEVERITY_INFO[before_sev]["label"],
            "color":    SEVERITY_INFO[before_sev]["color"],
            "confidence": before_conf,
        },
        "after": {
            "severity":   after_sev,
            "label":      SEVERITY_INFO[after_sev]["label"],
            "color":      SEVERITY_INFO[after_sev]["color"],
            "confidence": after_conf,
            "desc":       SEVERITY_INFO[after_sev]["desc"],
            "probs":      after_probs,
        },
        "damage_detected": damage_detected,
        "severity_change": diff,
        "result_image":    result_image_b64,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
