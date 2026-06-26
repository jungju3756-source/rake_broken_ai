# 제진기 부유물/레이크 고장 탐지 - 전이학습 & 검증 가이드

## 0. 환경
- GPU: RTX 4060 8GB / Python 3.10
- **학습용 GPU 토치 설치(중요, 현재는 CPU 토치임):**
  ```
  python -m pip uninstall -y torch torchvision
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- 검증: `python -c "import torch; print(torch.cuda.is_available())"` → True 여야 함

## 1. 영상 분석 요약
| 영상 | 내용 | 용도 |
|---|---|---|
| KakaoTalk_...mp4 (164s) | 스크린 전면 수면에 **부유물** 가득(비닐봉지·페트병·캔·나뭇가지·낙엽·담배꽁초) | **부유물 세그멘테이션** 데이터 |
| 오류 동영상.mp4 (83s) | 제진기/레이크(갈퀴) 동작·스크린 상태 변화 | **레이크 고장 탐지** 데이터 |

> ⚠️ COCO 사전학습 yolov8n-seg 를 그대로 돌리면 이 현장에서 거의 검출 안 됨(도메인 불일치).
> → 자체 라벨링 + 전이학습 필수.

## 2. 작업 순서
### (1) 라벨링용 프레임 추출
```
python 01_extract_frames.py --video "../KakaoTalk_20260620_121546018.mp4" --out frames/debris --every 12
python 01_extract_frames.py --video "../오류 동영상.mp4"               --out frames/rake   --every 12
```
### (2) 라벨링 (Roboflow 권장 / CVAT / labelme)
- 부유물: **폴리곤(세그멘테이션)** 라벨 → `data_debris_seg.yaml` 클래스 사용
- 레이크: **박스(detection)** 라벨 → `data_rake_det.yaml` 클래스 사용
- export 포맷: **YOLOv8** (images/ + labels/ + train/val 분할)
- 권장 분량: 각 작업 200~400장 이상, train:val = 8:2

### (3) 전이학습
```
python 03_train.py --task seg     # 부유물 세그멘테이션 (yolov8n-seg.pt 기반)
python 03_train.py --task det     # 레이크 고장 탐지   (yolov8n.pt 기반)
```
결과: `runs/seg/train/weights/best.pt`, `runs/det/train/weights/best.pt`

### (4) 카메라 입력 대용(동영상) 검출 검증
```
python 02_infer_video.py --model runs/seg/train/weights/best.pt --source "../KakaoTalk_20260620_121546018.mp4" --save out_debris.mp4 --show
python 02_infer_video.py --model runs/det/train/weights/best.pt --source "../오류 동영상.mp4"               --save out_rake.mp4   --show
```
실제 현장 카메라로 바꿀 때는 `--source 0` (또는 RTSP URL) 만 변경. **코드 수정 불필요.**

## 3. "다양한 방식" 전이학습 비교 실험 아이디어
1. **클래스 설계 2종**: (A) 부유물 다중클래스 vs (B) 단일 `debris` → mAP·라벨링 비용 비교
2. **freeze 비교**: `model.train(..., freeze=10)` (백본 동결, 데이터 적을 때) vs 전체 미세조정
3. **입력 해상도**: imgsz 640 vs 960 (작은 부유물 검출률 차이)
4. **모델 크기**: yolov8n-seg vs yolov8s-seg (속도 vs 정확도)
5. **레이크 고장 정의 3종**:
   - 상태 객체 탐지(rake_normal / rake_fault)  ← `data_rake_det.yaml` 접근법1
   - 부품 탐지 + 규칙 판정(debris_pile 잔류) ← 접근법2
   - 프레임 분류(yolov8n-cls, normal/fault)  ← `yolo classify train model=yolov8n-cls.pt`

## 4. 팁
- 8GB OOM 시 `--batch 8`
- 수면 반사/조도 변화 → `03_train.py`의 hsv 증강 유지
- 레이크 고장은 단일 프레임보다 **시간 흐름**이 중요 → 검출 후 "N초간 스크린에 협잡물 잔류 = 고장" 같은 후처리 규칙 추가 권장
