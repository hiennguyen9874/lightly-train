#!/usr/bin/env python3
"""
B1 - Pretrain DINOv2 ViT-B/14 on domain crops (person on motorbike).

No labels needed. Start from Meta pretrained weights (recommended for
small/medium domain datasets) and continue pretraining on your crops.

Edit the CONFIG block below, then run:
    python scripts/pretrain_dinov2_domain_vitb14.py
"""
import lightly_train

# ── CONFIG: sửa path tại đây ────────────────────────────────────────────────
DATA_DIR = "/home/jovyan/workspace/datasets/person-on-motorbike-images/"  # Thư mục chứa ảnh crop người ngồi trên xe máy.
OUT_DIR = "out/person-on-motorbike/dinov2_domain_vitb14"  # Thư mục output.
MODEL = "dinov2/vitb14"  # ViT-B/14 kèm weight Meta. Dùng "dinov2/vitb14-notpretrained" nếu muốn train từ scratch (cần >= ~1M ảnh).

# Ảnh crop ratio 2:1, chọn kích thước chia hết cho patch size 14.
# Dùng (252, 126) tránh lỗi reshape patch token của DINOv2 khi kích thước
# đầu vào không chia hết cho 14.
IMAGE_SIZE = (252, 126)  # (H, W)
WANDB_PROJECT = "dinov3-distillation"
WANDB_RUN_NAME = "dinov2_domain_vitb14"

# ── Pretrain ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lightly_train.pretrain(
        out=OUT_DIR,
        data=DATA_DIR,
        model=MODEL,
        method="dinov2",
        transform_args={
            "image_size": IMAGE_SIZE,
        },
        epochs="auto",  # Tự tính từ 125k steps. Data nhỏ -> epoch sẽ lớn, là bình thường.
        batch_size=128,  # GPU khỏe thì tăng lên 512-1024. Docs gốc khuyên ~3072 cho DINOv2.
        num_workers="auto",
        devices="auto",
        precision="bf16-mixed",  # Bật nếu GPU hỗ trợ bf16.
        # resume_interrupted=True,  # Bật khi muốn chạy tiếp run bị crash.
        # overwrite=False,
        loggers={
            "wandb": {
                "project": WANDB_PROJECT,
                "name": WANDB_RUN_NAME,
                "log_model": False,
            },
        },
    )
