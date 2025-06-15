import numpy as np
import cv2

def image_Grid(h, w, img, wall_thresh=210):
    """
    그레이스케일 맵(img)을 h×w 셀로 나눠,
    픽셀값 ≤ wall_thresh (검정·회색) : 벽(1)
    픽셀값  > wall_thresh (흰색)    : 길(0)
    """
    img_h, img_w = img.shape
    ch, cw = img_h // h, img_w // w

    grid = []
    for i in range(h):
        row = []
        for j in range(w):
            y1 = i*ch
            y2 = img_h if i==h-1 else (i+1)*ch
            x1 = j*cw
            x2 = img_w if j==w-1 else (j+1)*cw

            cell = img[y1:y2, x1:x2]
            wall_cnt = np.count_nonzero(cell <= wall_thresh)
            free_cnt = np.count_nonzero(cell >  wall_thresh)
            row.append(0 if free_cnt > wall_cnt else 1)
        grid.append(row)

    return grid, ch, cw

if __name__ == '__main__':
    PGM_FILE = '/home/seongunkim/grid/sample1.pgm'
    H, W     = 50, 50

    # 1) 맵 로드
    img = cv2.imread(PGM_FILE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load map: {PGM_FILE}")

    # 2) 그리드 생성
    grid, ch, cw = image_Grid(H, W, img, wall_thresh=210)

    # 3) free(0) 셀이 있는 영역만 잘라내기
    # 1) grid가 list-of-lists 형태라면 numpy 배열로 변환
    arr = np.array(grid, dtype=np.int8)

    # 2) 0 셀(=free) 위치들 찾기
    free_idxs = np.argwhere(arr == 0)

    if free_idxs.size == 0:
        print("free 셀이 없습니다.")
    else:
    # 3) 최소·최대 행·열 구하기
        r0, c0 = free_idxs.min(axis=0)
        r1, c1 = free_idxs.max(axis=0)

    # 4) 해당 부분만 잘라낸 subgrid
        subgrid = arr[r0:r1+1, c0:c1+1]

    # 5) 출력
        print(f"free 셀이 있는 부분만 잘라낸 sub-grid (rows {r0}–{r1}, cols {c0}–{c1}):")
        for row in subgrid:
            print(' '.join(str(int(v)) for v in row))
