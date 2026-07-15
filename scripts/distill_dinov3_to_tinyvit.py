#!/usr/bin/env python3
"""
Distillation: DINOv3 (teacher) → tinyvit (student).

Teacher: DINOv3 ViT-B/16 (dinov3/vitb16)
Student: TIMM tiny_vit_5m_224 (pretrained on ImageNet-1k with RandAugment)

Runs on unlabeled images — no labels needed.
"""
import lightly_train

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = "/home/jovyan/workspace/datasets/violence/data_hfps_tris_v1/processed/frames/train/"
OUT_DIR = "out/distill_dinov3_tinyvit"
TEACHER = "dinov3/vitb16"          # DINOv3 ViT-B/16 teacher
STUDENT = "timm/tiny_vit_5m_224"  # tinyvit student
WANDB_PROJECT = "dinov3-distillation"
WANDB_RUN_NAME = "dinov3-to-tinyvit"

# ── Pretrain ────────────────────────────────────────────────────────────────
lightly_train.pretrain(
    out=OUT_DIR,
    data=DATA_DIR,
    model=STUDENT,
    method="distillation",          # DistillationV3 (default): global + local loss
    method_args={
        "teacher": TEACHER,
    },
    epochs=100,                  # Auto-determine based on dataset size, "auto"
    batch_size=128,
    num_workers=4, # "auto"
    devices="auto",
    precision="bf16-mixed",
    overwrite=True,
    loggers={
        "wandb": {
            "project": WANDB_PROJECT,
            "name": WANDB_RUN_NAME,
            "log_model": False,
        },
    },
)
