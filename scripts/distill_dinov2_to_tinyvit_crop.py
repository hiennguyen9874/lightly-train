#!/usr/bin/env python3
"""
B2 - Distill ViT-in-domain (B1) -> TinyViT 5M on domain crops.

Student: timm/tiny_vit_5m_224.dist_in22k
Teacher: dinov2/vitb14 + teacher_weights từ B1 (exported_last.pt).
         Nếu chưa có B1, xóa "teacher_weights" để dùng weight Meta gốc.

Edit the CONFIG block below, then run:
    python scripts/distill_dinov2_to_tinyvit_crop.py
"""
import lightly_train

# ── CONFIG: sửa path tại đây ────────────────────────────────────────────────
DATA_DIR = "/home/jovyan/workspace/datasets/person-on-motorbike-images/"  # Cùng thư mục crop như B1.
OUT_DIR = "out/person-on-motorbike/tinyvit_from_dinov2_domain"  # Thư mục output B2.
STUDENT = "timm/tiny_vit_5m_224.dist_in22k"

TEACHER = "dinov2/vitb14"  # Phải cùng kiến trúc với model ở B1.
TEACHER_WEIGHTS = "out/person-on-motorbike/dinov2_domain_vitb14/exported_models/exported_last.pt"  # Output của B1. Set None nếu chưa có B1.
# TEACHER_WEIGHTS = None

# Giữ cùng size với B1. Kích thước ratio 2:1 này chia hết cho patch size
# 14 của teacher DINOv2, tránh lỗi reshape patch token.
IMAGE_SIZE = (252, 126)  # (H, W)

# ── Pretrain (distillation) ─────────────────────────────────────────────────
if __name__ == "__main__":
    method_args: dict = {"teacher": TEACHER}
    if TEACHER_WEIGHTS is not None:
        method_args["teacher_weights"] = TEACHER_WEIGHTS

    lightly_train.pretrain(
        out=OUT_DIR,
        data=DATA_DIR,
        model=STUDENT,
        method="distillation",  # Alias của distillationv3 (default).
        method_args=method_args,
        transform_args={
            "image_size": IMAGE_SIZE,
        },
        epochs=300,  # Data <100k crop có thể tăng lên 1000-3000.
        batch_size=128,
        num_workers="auto",
        devices="auto",
        # precision="bf16-mixed",  # Bật nếu GPU hỗ trợ bf16.
        # resume_interrupted=True,  # Bật khi muốn chạy tiếp run bị crash.
        # overwrite=False,
    )
