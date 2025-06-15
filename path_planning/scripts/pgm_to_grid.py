#!/usr/bin/env python3
import numpy as np
import cv2
import yaml
import argparse
import os

def image_Grid(h, w, img, wall_thresh=210, black_is_free=True):
    img_h, img_w = img.shape
    ch, cw = img_h // h, img_w // w

    grid = []
    for i in range(h):
        row = []
        for j in range(w):
            y1 = i * ch
            y2 = img_h if i == h - 1 else (i + 1) * ch
            x1 = j * cw
            x2 = img_w if j == w - 1 else (j + 1) * cw

            cell = img[y1:y2, x1:x2]
            if black_is_free:
                row.append(0 if cell.mean() < wall_thresh else 1)
            else:
                row.append(1 if cell.mean() < wall_thresh else 0)
        grid.append(row)

    return np.array(grid, dtype=np.uint8), ch, cw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgm', required=True, help='Path to .pgm image')
    parser.add_argument('--yaml', required=True, help='Path to .yaml file (for resolution)')
    parser.add_argument('--output', required=True, help='Path to save output .npy file')
    parser.add_argument('--height', type=int, default=50, help='Grid height')
    parser.add_argument('--width', type=int, default=50, help='Grid width')

    args, unknown = parser.parse_known_args()

    img = cv2.imread(args.pgm, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.pgm}")

    with open(args.yaml, 'r') as f:
        map_info = yaml.safe_load(f)
        resolution = map_info['resolution']

    grid, ch, cw = image_Grid(args.height, args.width, img)
    np.save(args.output, grid)
    print(f"[INFO] Grid map saved to {args.output} (cell size: {ch}x{cw} pixels)")
