# rake_broken_ai — 제진기 레이크 고장 감지 (YOLOv8 전이학습)

방학 캡스톤 디자인 프로젝트. 취수구 **제진기(bar screen)** 의 회전식 **레이크(갈퀴) 손상(고장)** 을
영상에서 자동 검출한다. YOLOv8n 객체탐지 모델을 전이학습(transfer learning)으로 도메인에 적응시킨 뒤,
영상을 **카메라 입력 대용**으로 넣어 고장 검출을 검증한다.

## 구성
```
capstone/
  01_extract_frames.py    # 동영상 → 프레임 추출 (라벨링 원본)
  04_select_diverse.py    # 중복 제거 / 다양한 프레임 선별 (과적합 방지)
  03_train.py             # yolov8n 전이학습 (GPU)
  02_infer_video.py       # 영상=카메라 입력 추론/검증
  models/
    rake_best.pt          # 1차 모델 (레이크 전체, 1클래스) — 참고용
    rake_broken_best.pt   # 2차 모델 (부서진 빗살, 2클래스) ★ 사용 모델
  전이학습_정리.md         # 전이학습 수행 전체 문서
  README_가이드.md
rake-fault-v2.v1i.yolov8/ # Roboflow 라벨 데이터셋 (rake / rake_broken)
```

## 빠른 시작
```bash
# 환경 (GPU)
pip install ultralytics opencv-python-headless
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 전이학습
python capstone/03_train.py --task det

# 영상으로 고장 검출 (카메라 대용)
python capstone/02_infer_video.py --model capstone/models/rake_broken_best.pt \
    --source "오류 동영상.mp4" --save out_rake_broken.mp4
# 실제 카메라
python capstone/02_infer_video.py --model capstone/models/rake_broken_best.pt --source 0 --show
```

## 결과 요약
- 2차 모델 `rake_broken`: mAP50 ≈ 0.456 (현실적 수치). 또렷한 손상은 conf 0.6+ 검출, 실시간 ~68 FPS.
- 자세한 과정·트러블슈팅: [`capstone/전이학습_정리.md`](capstone/전이학습_정리.md)

> 동영상/베이스 가중치/추출 프레임 등 대용량 파일은 `.gitignore`로 제외됨.
