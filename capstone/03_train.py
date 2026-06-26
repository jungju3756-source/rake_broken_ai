# -*- coding: utf-8 -*-
"""
03_train.py
yolov8n / yolov8n-seg 베이스 모델 전이학습 스크립트.
RTX 4060 (8GB) 기준 하이퍼파라미터.

부유물 세그멘테이션:
  python 03_train.py --task seg
레이크 고장 탐지:
  python 03_train.py --task det
"""
import argparse
from ultralytics import YOLO

CFG = {
    "seg": dict(base="yolov8n-seg.pt", data="data_debris_seg.yaml", project="runs/seg"),
    "det": dict(base="yolov8n.pt",     data="../rake-fault-v2.v1i.yolov8/data.yaml", project="runs/det"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["seg", "det"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)   # 8GB VRAM에서 안전. OOM시 8로.
    args = ap.parse_args()

    c = CFG[args.task]
    model = YOLO(c["base"])            # COCO 사전학습 가중치 → 전이학습
    model.train(
        data=c["data"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,                      # GPU
        project=c["project"],
        patience=30,                   # 30 epoch 개선 없으면 early stop
        # ---- 현장 영상 강건성을 위한 증강 ----
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # 조도/수면 반사 변화 대응
        degrees=5, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0,
        close_mosaic=10,               # 마지막 10 epoch은 mosaic 끔(미세조정)
    )
    print("학습 완료. best.pt:", c["project"])

if __name__ == "__main__":
    main()
