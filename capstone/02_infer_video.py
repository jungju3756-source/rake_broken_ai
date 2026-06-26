# -*- coding: utf-8 -*-
"""
02_infer_video.py
학습이 끝난(또는 사전학습) 모델로 동영상을 '카메라 입력 대용'으로 넣어 검출을 확인한다.
cv2.VideoCapture 는 카메라 인덱스(0,1,..)와 동영상 경로를 동일한 API로 처리하므로,
--source 0   -> 실제 USB/IP 카메라
--source xxx.mp4 -> 동영상 파일 (현장 카메라 대용)
완전히 동일한 코드로 동작한다.

사용 예 (부유물 세그멘테이션 모델):
  python 02_infer_video.py --model runs/seg/train/weights/best.pt --source "../KakaoTalk_20260620_121546018.mp4" --save out_debris.mp4
사용 예 (레이크 고장 탐지 모델):
  python 02_infer_video.py --model runs/det/train/weights/best.pt --source "../오류 동영상.mp4" --save out_rake.mp4
실제 카메라:
  python 02_infer_video.py --model best.pt --source 0 --show
"""
import argparse, time, cv2
from ultralytics import YOLO

def parse_source(s):
    # 숫자면 카메라 인덱스, 아니면 파일/스트림 경로
    return int(s) if s.isdigit() else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt 가중치 (학습 결과 best.pt 또는 사전학습 모델)")
    ap.add_argument("--source", required=True, help="카메라 인덱스(0) 또는 동영상 경로")
    ap.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", default="", help="결과 저장 mp4 경로 (빈값=저장안함)")
    ap.add_argument("--show", action="store_true", help="실시간 창 표시")
    ap.add_argument("--device", default="0", help="'0'=GPU, 'cpu'=CPU")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"소스를 열 수 없음: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 한 프레임 = 카메라 한 컷. 모델 추론.
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        vis = res.plot()  # 박스/마스크 오버레이된 BGR 이미지

        # 화면에 객체 수 표시 (부유물 개수 / 고장 탐지 여부 모니터링용)
        cv2.putText(vis, f"objects: {len(res.boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if writer:
            writer.write(vis)
        if args.show:
            cv2.imshow("YOLO - camera/video", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        n += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"완료: {n} 프레임, {dt:.1f}s, 평균 {n/max(dt,1e-6):.1f} FPS")
    if args.save:
        print(f"저장됨 → {args.save}")

if __name__ == "__main__":
    main()
