"""
Step 2: CNN 학습 (EfficientNet-B0 Transfer Learning)
=====================================================
데이터셋 폴더(dataset/train, dataset/val)를 읽어서
EfficientNet-B0 모델을 Fine-tuning합니다.

출력:
- best_model.pth  (가장 좋은 val accuracy 모델)
- training_log.png (학습 곡선 그래프)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time

# =====================================================================
# ✅ 경로 설정 — 본인 환경에 맞게 수정
# =====================================================================
DATASET_DIR = r"C:\Users\woain\Python_AI\car\dataset"
OUTPUT_DIR  = r"C:\Users\woain\Python_AI\car\model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 하이퍼파라미터
# =====================================================================
BATCH_SIZE  = 16
NUM_EPOCHS  = 30
LR          = 3e-4
NUM_CLASSES = 5
IMG_SIZE    = 224
PATIENCE    = 5       # Early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 사용 디바이스: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# =====================================================================
# 데이터 로더
# =====================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),  # ImageNet 기준
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"\n📦 데이터셋 로드 완료")
print(f"   train: {len(train_dataset)}개  |  val: {len(val_dataset)}개")
print(f"   클래스: {train_dataset.classes}")

# 클래스 불균형 대응 — 가중치 계산
class_counts = [len(os.listdir(os.path.join(DATASET_DIR, "train", str(i)))) for i in range(NUM_CLASSES)]
total = sum(class_counts)
class_weights = torch.tensor([total / (NUM_CLASSES * c) for c in class_counts], dtype=torch.float).to(DEVICE)
print(f"\n⚖️  클래스 가중치: {[round(w.item(), 2) for w in class_weights]}")

# =====================================================================
# 모델 정의 (EfficientNet-B0)
# =====================================================================
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # 마지막 분류기만 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model

model = build_model().to(DEVICE)
print(f"\n🧠 모델: EfficientNet-B0 (ImageNet pretrained)")
print(f"   출력 클래스: {NUM_CLASSES}개 (심각도 0~4)")

# =====================================================================
# 학습 설정
# =====================================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# =====================================================================
# 학습 함수
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total   += imgs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels

# =====================================================================
# 메인 학습 루프
# =====================================================================
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc  = 0.0
patience_cnt  = 0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

print(f"\n🚀 학습 시작! (epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LR})")
print("=" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - t0
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
          f"train loss: {train_loss:.4f}  acc: {train_acc:.3f}  |  "
          f"val loss: {val_loss:.4f}  acc: {val_acc:.3f}  |  "
          f"{elapsed:.1f}s")

    # Best 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_cnt = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, best_model_path)
        print(f"  💾 Best model 저장! val_acc={val_acc:.3f}")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping! (patience={PATIENCE})")
            break

print("=" * 60)
print(f"\n✅ 학습 완료! Best val accuracy: {best_val_acc:.3f}")

# =====================================================================
# 최종 평가 (best model 로드)
# =====================================================================
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint["model_state_dict"])
_, final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion)

label_names = ["0-정상", "1-경미", "2-보통", "3-심각", "4-매우심각"]
print(f"\n📊 최종 분류 리포트 (val set):")
print(classification_report(final_labels, final_preds, target_names=label_names))

# =====================================================================
# 학습 곡선 저장
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="train")
ax1.plot(history["val_loss"],   label="val")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history["train_acc"], label="train")
ax2.plot(history["val_acc"],   label="val")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
log_path = os.path.join(OUTPUT_DIR, "training_log.png")
plt.savefig(log_path, dpi=150)
plt.show()
print(f"\n📈 학습 곡선 저장: {log_path}")
print(f"🏁 모델 저장 위치: {best_model_path}")
print(f"\n➡️  다음 단계: step3_pipeline.py 실행")
