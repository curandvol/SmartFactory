#!/usr/bin/env python3
import argparse
import numpy as np
import cv2
# 디버깅용 파일 (grid 화 한것을 다시 PGM 파일로 계산)
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True, help='Path to grid.npy')
    p.add_argument('--output', required=True, help='Path to write .pgm')
    p.add_argument('--scale',  type=int, default=4,
                   help='How many pixels per grid cell (for visibility)')
                   
    p.add_argument('--ref', help='(optional) Path to original pgm for exact resolution')
    args = p.parse_args()

    # 1) load numpy grid
    grid = np.load(args.input)  # dtype should be 0/1
    # invert & scale to 0~255
    img = (1 - grid) * 255      # free→255, wall→0
    img = img.astype(np.uint8)

    # 2) 원본 해상도 유지 or 기존 scale 대로 확대
    if args.ref:
        # 원본 PGM 불러와 크기 그대로 사용
        orig = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
        if orig is None:
            raise FileNotFoundError(f"Cannot load reference PGM: {args.ref}")
        H, W = orig.shape
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    elif args.scale != 1:
        h, w = img.shape
        img = cv2.resize(img, (w * args.scale, h * args.scale), interpolation=cv2.INTER_NEAREST)

    # 3) 저장 (OpenCV will infer PGM from extension)
    cv2.imwrite(args.output, img)
    print(f"Saved PGM to {args.output}, shape={img.shape}")

if __name__ == '__main__':
    main()
