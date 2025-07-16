#!/usr/bin/env python3
import argparse
import numpy as np
import cv2
# 디버깅용 파일 (grid 화 시킨것을 PGM 파일로 계산)
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True, help='Path to grid.npy')
    p.add_argument('--output', required=True, help='Path to write .pgm')
    p.add_argument('--scale',  type=int, default=4,
                   help='How many pixels per grid cell (for visibility)')
    args = p.parse_args()

    # 1) load numpy grid
    grid = np.load(args.input)  # dtype should be 0/1
    # invert & scale to 0~255
    img = (1 - grid) * 255      # free→255, wall→0
    img = img.astype(np.uint8)

    # 2) optional: 각 셀을 더 크게(가시성 위해) 확대
    if args.scale != 1:
        h, w = img.shape
        img = cv2.resize(img,
                         (w * args.scale, h * args.scale),
                         interpolation=cv2.INTER_NEAREST)

    # 3) 저장 (OpenCV will infer PGM from extension)
    cv2.imwrite(args.output, img)
    print(f"Saved PGM to {args.output}, shape={img.shape}")

if __name__ == '__main__':
    main()
