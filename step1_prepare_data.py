"""
Step 1: ROI 패치 추출 + 데이터 증강
====================================
before/after 이미지 쌍에서 OpenCV로 흠집 영역(ROI) 패치를 추출하고,
심각도 라벨을 붙여 CNN 학습용 데이터셋을 만듭니다.

출력 폴더 구조:
dataset/
├── train/
│   ├── 0/  (정상)
│   ├── 1/  (경미)
│   ├── 2/  (보통)
│   ├── 3/  (심각)
│   └── 4/  (매우 심각)
└── val/
    ├── 0/
    └── ...
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# =====================================================================
# ✅ 경로 설정 — 본인 환경에 맞게 수정
# =====================================================================
BEFORE_DIR = r"C:\Users\woain\Python_AI\car\before"
AFTER_DIR  = r"C:\Users\woain\Python_AI\car\after"
LABEL_CSV  = r"C:\Users\woain\Python_AI\car\labels.csv"   # car_id, score
OUTPUT_DIR = r"C:\Users\woain\Python_AI\car\dataset"

# CSV가 없을 경우 여기에 직접 입력 (car_id: score)
MANUAL_LABELS = {
    # "car001": 0,
    # "car002": 2,
    # 필요하면 여기에 추가
}

ANGLES = ["front", "left", "left_side", "rear", "right", "right_side"]

# =====================================================================
# 파라미터
# =====================================================================
PATCH_SIZE   = 224          # CNN 입력 크기 (EfficientNet/ResNet 표준)
VAL_RATIO    = 0.3          # 검증셋 비율
AUG_PER_PATCH = 5           # 패치 1개당 증강 이미지 수
MIN_PATCH_AREA = 60 * 60    # 너무 작은 패치 제외 (px²)

# OpenCV 파라미터 (기존 코드 기반)
REF_W, REF_H   = 1920, 1080
ROI_REF        = (100, 100, 1800, 950)
USE_ECC        = True
ECC_ITERS      = 60
ECC_EPS        = 1e-5
PCTL           = 97.0
ABS_THR        = 4
MIN_AREA       = 3
MIN_ASPECT     = 1.2
MAX_ASPECT     = 400.0
MIN_FILL       = 0.001
MIN_SCORE      = 2.0
MAX_BOX_AREA_RATIO = 0.05
LINE_LEN       = 41
LINE_PCTL      = 99.2
LINE_MIN       = 3
TOP_CUT_RATIO  = 0.35

# =====================================================================
# OpenCV 유틸 함수 (기존 damege_box.ipynb 기반)
# =====================================================================
def preprocess_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)

def scale_roi(w, h):
    x1, y1, x2, y2 = ROI_REF
    rx1 = max(0, int(x1 / REF_W * w))
    ry1 = max(0, int(y1 / REF_H * h))
    rx2 = min(w, int(x2 / REF_W * w))
    ry2 = min(h, int(y2 / REF_H * h))
    return (rx1, ry1, rx2, ry2)

def align_ecc(ref_g, mov_g, roi):
    x1, y1, x2, y2 = roi
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(
            ref_g[y1:y2, x1:x2], mov_g[y1:y2, x1:x2],
            warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
        )
        return cv2.warpAffine(mov_g, warp, (ref_g.shape[1], ref_g.shape[0]),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except Exception:
        return mov_g

def detect_boxes(bgr_before, bgr_after):
    """before/after 쌍에서 흠집 박스 목록 반환"""
    h, w = bgr_after.shape[:2]
    roi = scale_roi(w, h)
    x1, y1, x2, y2 = roi

    gb = preprocess_gray(bgr_before)
    ga = preprocess_gray(bgr_after)
    if USE_ECC:
        ga = align_ecc(gb, ga, roi)


    # 크기 불일치 방지 — after를 before 크기로 맞춤
    if ga.shape != gb.shape:
        ga = cv2.resize(ga, (gb.shape[1], gb.shape[0]))
    
    diff = cv2.absdiff(ga, gb)
    thr  = max(float(np.percentile(diff[y1:y2, x1:x2], PCTL)), ABS_THR)

    mask = np.zeros_like(diff)
    roi_mask = (diff[y1:y2, x1:x2] >= thr).astype(np.uint8) * 255
    mask[y1:y2, x1:x2] = roi_mask

    # 상단 컷
    cut = int(y1 + TOP_CUT_RATIO * (y2 - y1))
    mask[y1:cut, x1:x2] = 0

    # 라인 강조
    L  = (LINE_LEN | 1)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    line = cv2.max(
        cv2.morphologyEx(diff, cv2.MORPH_TOPHAT,  k1),
        cv2.morphologyEx(diff, cv2.MORPH_TOPHAT,  k2),
    )
    t = float(np.percentile(line[y1:y2, x1:x2], LINE_PCTL))
    mask_line = np.zeros_like(mask)
    mask_line[y1:y2, x1:x2] = (line[y1:y2, x1:x2] >= max(t, LINE_MIN)).astype(np.uint8) * 255
    mask_line = cv2.bitwise_and(mask_line, mask)
    mask = cv2.bitwise_or(mask, mask_line)

    # 모폴로지
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))

    # 컨투어 → 박스
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    roi_area = max(1, (x2-x1)*(y2-y1))

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = max(bw/max(1,bh), bh/max(1,bw))
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        if (bw*bh) > MAX_BOX_AREA_RATIO * roi_area:
            continue

        fill = float((mask[by:by+bh, bx:bx+bw] > 0).mean())
        if fill < MIN_FILL:
            continue

        patch_diff = cv2.absdiff(ga[by:by+bh, bx:bx+bw], gb[by:by+bh, bx:bx+bw])
        patch_m    = mask[by:by+bh, bx:bx+bw] > 0
        vals = patch_diff[patch_m]
        if vals.size == 0:
            continue
        score = float(np.percentile(vals, 90))
        if score < MIN_SCORE:
            continue

        boxes.append((bx, by, bx+bw, by+bh))

    return boxes

# =====================================================================
# 데이터 증강
# =====================================================================
def augment(patch):
    """패치 1장 → AUG_PER_PATCH장 반환"""
    results = [patch]  # 원본 포함

    # 좌우 반전
    results.append(cv2.flip(patch, 1))

    # 회전 (±15도)
    h, w = patch.shape[:2]
    cx, cy = w//2, h//2
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        results.append(cv2.warpAffine(patch, M, (w, h),
                       borderMode=cv2.BORDER_REFLECT))

    # 밝기 변화
    for alpha in [0.8, 1.2]:
        bright = np.clip(patch.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        results.append(bright)

    # 가우시안 블러 (약한 노이즈 대체)
    results.append(cv2.GaussianBlur(patch, (3, 3), 0))

    # 상하 반전
    results.append(cv2.flip(patch, 0))

    # AUG_PER_PATCH 수만큼만 반환
    return results[:AUG_PER_PATCH]

# =====================================================================
# 라벨 로드
# =====================================================================
def load_labels():
    labels = dict(MANUAL_LABELS)
    if os.path.exists(LABEL_CSV):
        df = pd.read_csv(LABEL_CSV)
        # 컬럼명 자동 인식 (car_id / score 또는 유사한 이름)
        id_col    = [c for c in df.columns if 'id' in c.lower()][0]
        score_col = [c for c in df.columns if 'score' in c.lower() or 'label' in c.lower() or 'severity' in c.lower()][0]
        for _, row in df.iterrows():
            car_id = str(row[id_col]).strip()
            labels[car_id] = int(row[score_col])
        print(f"✅ CSV 로드 완료: {len(labels)}개 라벨")
    else:
        print(f"⚠️  CSV 없음 — MANUAL_LABELS 사용 ({len(labels)}개)")
    return labels

# =====================================================================
# 메인
# =====================================================================
def main():
    labels = load_labels()
    if not labels:
        print("❌ 라벨이 없습니다. LABEL_CSV 경로 또는 MANUAL_LABELS를 확인하세요.")
        return

    # 출력 폴더 생성
    for split in ["train", "val"]:
        for cls in range(5):
            os.makedirs(os.path.join(OUTPUT_DIR, split, str(cls)), exist_ok=True)

    stats = {"total_pairs": 0, "patches_extracted": 0, "patches_saved": 0, "skipped": 0}
    all_patches = []  # (patch_img, label) 리스트 — 나중에 train/val 분리

    # 차량 ID 수집
    car_ids = sorted(labels.keys())
    print(f"\n🚗 처리할 차량: {len(car_ids)}대\n")

    for car_id in car_ids:
        severity = labels[car_id]
        car_patches = []

        for angle in ANGLES:
            # 파일 찾기 (png/jpg 모두 지원)
            b_path = None
            a_path = None
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
                bp = os.path.join(BEFORE_DIR, f"{car_id}_{angle}{ext}")
                ap = os.path.join(AFTER_DIR,  f"{car_id}_{angle}{ext}")
                if os.path.exists(bp) and os.path.exists(ap):
                    b_path, a_path = bp, ap
                    break

            if b_path is None:
                continue

            bgr_b = cv2.imread(b_path)
            bgr_a = cv2.imread(a_path)
            if bgr_b is None or bgr_a is None:
                continue

            stats["total_pairs"] += 1
            boxes = detect_boxes(bgr_b, bgr_a)

            if not boxes:
                # 패치 미감지 → severity 상관없이 after 전체 이미지 사용
                patch = cv2.resize(bgr_a, (PATCH_SIZE, PATCH_SIZE))
                car_patches.append(patch)
                continue

            for (bx1, by1, bx2, by2) in boxes:
                pw, ph = bx2-bx1, by2-by1
                if pw*ph < MIN_PATCH_AREA:
                    continue

                # 패딩 추가 (20%)
                pad_w = int(pw * 0.2)
                pad_h = int(ph * 0.2)
                H, W  = bgr_a.shape[:2]
                bx1p  = max(0, bx1-pad_w)
                by1p  = max(0, by1-pad_h)
                bx2p  = min(W, bx2+pad_w)
                by2p  = min(H, by2+pad_h)

                patch = bgr_a[by1p:by2p, bx1p:bx2p]
                patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                car_patches.append(patch)
                stats["patches_extracted"] += 1

        if not car_patches:
            print(f"  [{car_id}] ⚠️  패치 없음 — 스킵")
            continue

        all_patches.extend([(p, severity) for p in car_patches])
        print(f"  [{car_id}] severity={severity}  패치 {len(car_patches)}개 추출")

    # ── train/val 분리 + 증강 저장 ──
    print(f"\n📦 총 원본 패치: {len(all_patches)}개")
    print(f"🔀 train/val 분리 중... (val {int(VAL_RATIO*100)}%)")

    np.random.seed(42)
    indices  = np.random.permutation(len(all_patches))
    val_size = int(len(all_patches) * VAL_RATIO)
    val_idx  = set(indices[:val_size])

    saved_count = {"train": 0, "val": 0}

    for i, (patch, label) in enumerate(all_patches):
        split = "val" if i in val_idx else "train"

        if split == "train":
            imgs_to_save = augment(patch)   # 증강 적용
        else:
            imgs_to_save = [patch]          # val은 원본만

        for j, img in enumerate(imgs_to_save):
            fname = f"patch_{i:05d}_aug{j}.png"
            out   = os.path.join(OUTPUT_DIR, split, str(label), fname)
            cv2.imwrite(out, img)
            saved_count[split] += 1
            stats["patches_saved"] += 1

    # ── 결과 요약 ──
    print("\n" + "="*50)
    print("✅ 데이터 준비 완료!")
    print(f"   처리된 쌍:      {stats['total_pairs']}개")
    print(f"   추출된 패치:    {stats['patches_extracted']}개")
    print(f"   저장된 이미지:  {stats['patches_saved']}개")
    print(f"   - train:        {saved_count['train']}개 (증강 포함)")
    print(f"   - val:          {saved_count['val']}개")
    print(f"   스킵:           {stats['skipped']}개")
    print("="*50)

    # 클래스별 분포 출력
    print("\n📊 클래스별 분포 (train):")
    label_names = {0:"정상", 1:"경미", 2:"보통", 3:"심각", 4:"매우심각"}
    for cls in range(5):
        folder = os.path.join(OUTPUT_DIR, "train", str(cls))
        n = len(os.listdir(folder))
        bar = "█" * (n // 10)
        print(f"   [{cls}] {label_names[cls]:6s}: {n:4d}개  {bar}")

    print(f"\n➡️  다음 단계: step2_train_cnn.py 실행")

if __name__ == "__main__":
    main()
