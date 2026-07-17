#!/usr/bin/env python3
"""Distill DINOv3 ViT-L/16 into an Ultralytics YOLO26s backbone.

The dataset is unlabeled: LightlyTrain reads JPEGs recursively from ``DATA_DIR``.
The exported student checkpoint can be fine-tuned for object detection with Ultralytics.

Install the optional Ultralytics dependency before running this script::

    pip install "lightly-train[ultralytics]"
"""

from __future__ import annotations

import lightly_train

DATA_DIR = "/home/jovyan/workspace/datasets/traffic-images"
OUT_DIR = "out/distill_dinov3_vitl16_yolo26s"
TEACHER = "dinov3/vitl16"
STUDENT = "ultralytics/yolo26s.yaml"
WANDB_PROJECT = "dinov3-distillation"
WANDB_RUN_NAME = "dinov3-to-yolo26s"

# ViT-L/16 is substantially larger than the students in the other examples, so use a
# batch size that is less likely to exceed GPU memory. Increase it when memory allows.
EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 8


def main() -> None:
    """Run DINOv3-to-YOLO26s distillation on the traffic image dataset."""
    lightly_train.pretrain(
        out=OUT_DIR,
        data=DATA_DIR,
        model=STUDENT,
        method="distillation",
        method_args={"teacher": TEACHER},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        devices="auto",
        precision="bf16-mixed",
        resume_interrupted=True,
    loggers={
        "wandb": {
            "project": WANDB_PROJECT,
            "name": WANDB_RUN_NAME,
            "log_model": False,
        },
    },
    )


if __name__ == "__main__":
    main()
