# -*- coding: utf-8 -*-
"""
04_select_diverse.py
동영상에서 '서로 충분히 다른' 프레임만 골라 저장 (중복/거의같은프레임 제거).
라벨링 효율을 위해 다양한 장면만 추린다.

사용 예:
  python 04_select_diverse.py --video "../오류 동영상.mp4" --out frames/rake_selected --target 60 --prefix rake
"""
import argparse, os, cv2, numpy as np

def imwrite_unicode(path, img, q=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok: buf.tofile(path)
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=60, help="목표 장수(근사)")
    ap.add_argument("--prefix", default="frame")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"열 수 없음: {args.video}")

    # 1) 모든 프레임의 축소 그레이스케일 시그니처 수집
    sigs, frames_idx = [], []
    idx = 0
    raw = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        small = cv2.cvtColor(cv2.resize(fr, (64, 36)), cv2.COLOR_BGR2GRAY)
        sigs.append(small.astype(np.float32))
        frames_idx.append(idx)
        idx += 1
    total = len(sigs)

    # 2) 인접 프레임 차이 기반으로 '변화량'이 큰 프레임 우선 선택
    #    탐욕적으로: 이미 고른 것들과의 최소 차이가 큰 프레임을 target개 선택
    chosen = [0]  # 첫 프레임
    diffs_to_chosen = [np.mean(np.abs(sigs[i] - sigs[0])) for i in range(total)]
    while len(chosen) < min(args.target, total):
        # 현재까지 고른 집합과 가장 '다른' 프레임 선택
        k = int(np.argmax(diffs_to_chosen))
        if diffs_to_chosen[k] <= 0:
            break
        chosen.append(k)
        # 갱신: 각 프레임의 '가장 가까운 선택프레임과의 거리'를 유지
        for i in range(total):
            d = np.mean(np.abs(sigs[i] - sigs[k]))
            if d < diffs_to_chosen[i]:
                diffs_to_chosen[i] = d

    chosen = sorted(set(chosen))

    # 3) 실제 프레임 다시 읽어 저장
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    want = set(chosen); saved = 0; i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i in want:
            imwrite_unicode(os.path.join(args.out, f"{args.prefix}_{i:06d}.jpg"), fr)
            saved += 1
        i += 1
    cap.release()
    print(f"전체 {total}프레임 → 다양한 {saved}장 선택 저장: {args.out}")

if __name__ == "__main__":
    main()
