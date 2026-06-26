# -*- coding: utf-8 -*-
"""
01_extract_frames.py
동영상에서 라벨링용 프레임을 일정 간격으로 추출한다.
- 부유물(segmentation)용 / 레이크 고장(detection)용 데이터셋의 '원본 이미지'를 만드는 단계.
- 추출된 이미지는 Roboflow / CVAT / labelImg 등에서 라벨링한다.

사용 예:
  python 01_extract_frames.py --video "../KakaoTalk_20260620_121546018.mp4" --out frames/debris --every 12
  python 01_extract_frames.py --video "../오류 동영상.mp4" --out frames/rake --every 12
"""
import argparse, os, cv2, numpy as np

def imwrite_unicode(path, img, quality=92):
    # Windows에서 cv2.imwrite는 한글/유니코드 경로에 저장 실패 → imencode로 우회
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)   # numpy.tofile 은 유니코드 경로 지원
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="입력 동영상 경로")
    ap.add_argument("--out", required=True, help="프레임 저장 폴더")
    ap.add_argument("--every", type=int, default=12, help="N프레임마다 1장 저장 (24fps에서 12=초당 2장)")
    ap.add_argument("--max", type=int, default=0, help="최대 저장 장수 (0=제한없음)")
    ap.add_argument("--prefix", default="", help="출력 파일명 접두사 (한글 영상명 회피용, 예: rake)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"동영상을 열 수 없음: {args.video}")

    # 출력 파일명은 ASCII로 강제(한글 영상명이면 --prefix 사용 권장)
    base = args.prefix or os.path.splitext(os.path.basename(args.video))[0]
    if not base.isascii():
        base = "frame"
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            path = os.path.join(args.out, f"{base}_{idx:06d}.jpg")
            imwrite_unicode(path, frame, 92)
            saved += 1
            if args.max and saved >= args.max:
                break
        idx += 1
    cap.release()
    print(f"완료: {saved}장 저장 → {args.out}")

if __name__ == "__main__":
    main()
