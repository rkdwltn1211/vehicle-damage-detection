"""
Step 3: Before/After CNN 심각도 비교 파이프라인
=================================================
before/after 이미지 쌍을 각각 CNN에 넣어
심각도 변화를 비교하고 결과를 시각화합니다.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# =====================================================================
# ✅ 경로 설정
# =====================================================================
MODEL_PATH = r"C:\Users\woain\Python_AI\car\model\best_model.pth"
BEFORE_DIR = r"C:\Users\woain\Python_AI\car\before"
AFTER_DIR  = r"C:\Users\woain\Python_AI\car\after"
OUTPUT_DIR = r"C:\Users\woain\Python_AI\car\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
IMG_SIZE    = 224
ANGLES      = ["front", "left", "left_side", "rear", "right", "right_side"]

SEVERITY_LABELS = {
    0: ("Normal",   (34, 197, 94)),
    1: ("Minor",    (250, 204, 21)),
    2: ("Moderate", (249, 115, 22)),
    3: ("Severe",   (239, 68, 68)),
    4: ("Critical", (153, 27, 27)),
}

CHANGE_THRESHOLD = 2  # Lv.2 이상일 때만 흠집으로 판정


# =====================================================================
# CNN 모델 로드
# =====================================================================
def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    checkpoint = torch.load(model_path, map_location=DEVICE)
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

def predict_severity(model, bgr_img):
    rgb    = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    tensor = cnn_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    return int(pred.item()), float(conf.item())

# =====================================================================
# 결과 패널 생성
# =====================================================================
def make_result_panel(bgr_before, bgr_after,
                      before_sev, before_conf,
                      after_sev, after_conf,
                      car_id, angle):

    PANEL_W = 820
    PANEL_H = 480
    IMG_W   = 360
    IMG_H   = 270
    MARGIN  = 20

    panel = np.ones((PANEL_H, PANEL_W, 3), dtype=np.uint8) * 25

    # 제목
    cv2.putText(panel, f"[{car_id} | {angle}]  Damage Assessment",
                (MARGIN, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210,210,210), 2)
    cv2.line(panel, (MARGIN, 42), (PANEL_W-MARGIN, 42), (70,70,70), 1)

    # Before 이미지
    b_res = cv2.resize(bgr_before, (IMG_W, IMG_H))
    panel[55:55+IMG_H, MARGIN:MARGIN+IMG_W] = b_res
    b_label, b_color = SEVERITY_LABELS[before_sev]
    cv2.rectangle(panel, (MARGIN, 55+IMG_H), (MARGIN+IMG_W, 55+IMG_H+28), (45,45,45), -1)
    cv2.putText(panel, f"BEFORE: {b_label}  (Lv.{before_sev} | {before_conf*100:.0f}%)",
                (MARGIN+6, 55+IMG_H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 1)

    # After 이미지
    a_res = cv2.resize(bgr_after, (IMG_W, IMG_H))
    ax = PANEL_W - MARGIN - IMG_W
    panel[55:55+IMG_H, ax:ax+IMG_W] = a_res
    a_label, a_color = SEVERITY_LABELS[after_sev]
    cv2.rectangle(panel, (ax, 55+IMG_H), (ax+IMG_W, 55+IMG_H+28), (45,45,45), -1)
    cv2.putText(panel, f"AFTER:  {a_label}  (Lv.{after_sev} | {after_conf*100:.0f}%)",
                (ax+6, 55+IMG_H+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, a_color, 1)

    # 화살표
    mid_x = PANEL_W // 2
    mid_y = 55 + IMG_H // 2
    cv2.arrowedLine(panel, (mid_x-28, mid_y), (mid_x+28, mid_y),
                    (130,130,130), 2, tipLength=0.4)

    # 심각도 변화
    diff = after_sev - before_sev

    change_y = 55 + IMG_H + 60

    

    if after_sev >= CHANGE_THRESHOLD and diff > 0:
        change_text  = f"Severity INCREASED  +{diff} level"
        change_color = (80, 80, 240)
    elif diff < 0:
        change_text  = f"Severity DECREASED  {diff} level"
        change_color = (80, 200, 80)
    else:
        change_text  = "No Change Detected"
        change_color = (160, 160, 160)

    cv2.rectangle(panel, (MARGIN, change_y-26), (PANEL_W-MARGIN, change_y+8), (45,45,45), -1)
    cv2.putText(panel, change_text,
                (PANEL_W//2 - 170, change_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, change_color, 2)

    # 심각도 바
    bar_y  = change_y + 40
    bar_w  = (PANEL_W - MARGIN*2) // 5
    colors = [(34,197,94),(250,204,21),(249,115,22),(239,68,68),(153,27,27)]
    cv2.putText(panel, "Severity Scale:", (MARGIN, bar_y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1)

    for i in range(5):
        bx1 = MARGIN + i * bar_w
        bx2 = bx1 + bar_w - 4
        by1, by2 = bar_y, bar_y + 24
        col = colors[i] if i == after_sev else (65,65,65)
        cv2.rectangle(panel, (bx1, by1), (bx2, by2), col, -1)
        if i == after_sev:
            cv2.rectangle(panel, (bx1-2, by1-2), (bx2+2, by2+2), colors[i], 2)
        cv2.putText(panel, str(i), (bx1+bar_w//2-5, by1+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255,255,255) if i == after_sev else (100,100,100), 1)

    return panel


def find_image(folder, car_id, angle):
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        p = os.path.join(folder, f"{car_id}_{angle}{ext}")
        if os.path.exists(p):
            return p
    return None


# =====================================================================
# 메인
# =====================================================================
def main():
    print("🚗 탁송 차량 흠집 비교 분석 시작\n")
    model = load_model(MODEL_PATH)

    # ✅ 테스트할 차량 ID
    TEST_CARS = ["car076", "car086", "car098", "car041", "car001"]

    summary = []

    for car_id in TEST_CARS:
        print(f"\n── {car_id} 분석 중...")
        car_before_sevs = []
        car_after_sevs  = []

        for angle in ANGLES:
            b_path = find_image(BEFORE_DIR, car_id, angle)
            a_path = find_image(AFTER_DIR,  car_id, angle)
            if not b_path or not a_path:
                continue

            bgr_b = cv2.imread(b_path)
            bgr_a = cv2.imread(a_path)
            if bgr_b is None or bgr_a is None:
                continue

            before_sev, before_conf = 0, 1.0
            after_sev,  after_conf  = predict_severity(model, bgr_a)

            car_before_sevs.append(before_sev)
            car_after_sevs.append(after_sev)

            panel = make_result_panel(
                bgr_b, bgr_a,
                before_sev, before_conf,
                after_sev, after_conf,
                car_id, angle
            )
            out_path = os.path.join(OUTPUT_DIR, f"{car_id}_{angle}_compare.jpg")
            cv2.imwrite(out_path, panel)

            b_label = SEVERITY_LABELS[before_sev][0]
            a_label = SEVERITY_LABELS[after_sev][0]
            diff    = after_sev - before_sev
            change  = f"+{diff}" if diff > 0 else str(diff)
            print(f"   [{angle:12s}] Before: {b_label:8s}(Lv.{before_sev}) -> After: {a_label:8s}(Lv.{after_sev})  ({change})")

        if car_before_sevs and car_after_sevs:
            final_before = max(car_before_sevs)
            final_after  = max(car_after_sevs)
            diff    = final_after - final_before
            change  = f"+{diff}" if diff > 0 else str(diff)
            verdict = "WARNING: Damage Increased!" if diff > 0 else ("OK: No Change" if diff == 0 else "OK: Improved")
            print(f"\n   {'='*50}")
            print(f"   [{car_id}] Before Lv.{final_before} -> After Lv.{final_after}  ({change})  {verdict}")
            summary.append((car_id, final_before, final_after, diff))

    # 전체 요약
    print(f"\n{'='*60}")
    print("📊 전체 분석 요약")
    print(f"{'='*60}")
    print(f"{'차량ID':<10} {'Before':>8} {'After':>8} {'변화':>6}  판정")
    print(f"{'-'*60}")
    for car_id, b, a, d in summary:
        change  = f"+{d}" if d > 0 else str(d)
        verdict = "⚠️  흠집 증가" if d > 0 else ("✅ 이상없음" if d == 0 else "✅ 개선됨")
        print(f"{car_id:<10} {'Lv.'+str(b):>8} {'Lv.'+str(a):>8} {change:>6}  {verdict}")

    print(f"\n📁 결과 저장: {OUTPUT_DIR}")
    print("➡️  다음 단계: FastAPI 연결")

if __name__ == "__main__":
    main()
