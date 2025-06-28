#!/usr/bin/env python3
import numpy as np
import cv2
import yaml
import argparse
import os

def image_Grid(h, w, img, wall_thresh=210):
    """
    이미지 데이터를 grid map으로 변환

    h: grid의 높이 셀 개수
    w: grid의 너비 셀 개수
    img: 입력 이미지 (grayscale)
    wall_thresh: 밝기값 임계치 
    """
    img_h, img_w = img.shape  # 이미지의 세로, 가로 픽셀 크기
    ch, cw = img_h // h, img_w // w  # grid 셀 하나당 픽셀 크기

    grid = []  # 최종 grid 데이터를 담을 리스트

    for i in range(h):  # 각 row 순회
        row = []
        for j in range(w):  # 각 column 순회
            # 셀에 해당하는 이미지 영역 좌표 계산
            y1 = i * ch
            y2 = img_h if i == h - 1 else (i + 1) * ch
            x1 = j * cw
            x2 = img_w if j == w - 1 else (j + 1) * cw

            # 셀 영역 잘라오기
            cell = img[y1:y2, x1:x2]

            # 검은색 = wall → 평균 밝기 < 임계치 → wall(1), 아니면 free(0)
            row.append(1 if cell.mean() < wall_thresh else 0)
        grid.append(row)  # 완성된 row를 grid에 추가

    # numpy array로 변환하고 셀 크기 정보 함께 반환
    return np.array(grid, dtype=np.uint8), ch, cw

if __name__ == '__main__':
    # 명령줄에서 입력된 옵션값 읽기 위해 객체 등록
    parser = argparse.ArgumentParser()

    # 필수 인자 등록
    parser.add_argument('--pgm', required=True, help='Path to .pgm image')  # .pgm 이미지 경로
    parser.add_argument('--yaml', required=True, help='Path to .yaml file (for resolution)')  # .yaml 파일 경로
    parser.add_argument('--output', required=True, help='Path to save output .npy file')  # 출력 .npy 경로

    # 인자 파싱
    args, _ = parser.parse_known_args()

    # .pgm 이미지 읽기
    img = cv2.imread(args.pgm, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.pgm}")

    # .yaml 파일 읽기 (resolution만 읽음)
    with open(args.yaml, 'r') as f:
        map_info = yaml.safe_load(f)
        resolution = map_info['resolution'] 

    # 로봇 크기 (m) + 여유
    robot_width = 0.55  # 로봇 폭 550mm = 0.55m
    safety_margin = 0.05  # 5cm 여유
    cell_size_m = (robot_width / 2) + safety_margin

    # pixel 단위 cell 크기
    cell_size_pix = cell_size_m / resolution

    # grid 크기 계산
    grid_h = max(1, int(img.shape[0] / cell_size_pix))
    grid_w = max(1, int(img.shape[1] / cell_size_pix))

    print(f"[INFO] Robot width: {robot_width}m, Cell size: {cell_size_m:.2f}m")
    print(f"[INFO] Auto grid size: {grid_h} x {grid_w} (cell ~{cell_size_pix:.2f} px)")

    grid, ch, cw = image_Grid(grid_h, grid_w, img)

    print("[INFO] Grid map (0=free, 1=wall):")
    for row in grid:
        print(' '.join(str(cell) for cell in row))

    np.save(args.output, grid)
    print(f"[INFO] Grid map saved to {args.output} (cell size: {ch}x{cw} pixels)")
