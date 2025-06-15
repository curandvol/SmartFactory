import yaml
import os
from PIL import Image
import numpy as np

def map_to_coarse_grid(yaml_file_path, pgm_file_path,
                       chunk_size,
                       non_navigable_threshold_ratio,
                       wall_threshold_ratio=0.05):
    with open(yaml_file_path, 'r') as f:
        yaml.safe_load(f)
    raw = np.array(Image.open(pgm_file_path))
    h, w = raw.shape
    ch, cw = h // chunk_size, w // chunk_size
    grid = np.zeros((ch, cw), dtype=np.uint8)  # 0: non-nav, 1: free, 2: wall

    for r in range(ch):
        for c in range(cw):
            blk = raw[r*chunk_size:(r+1)*chunk_size,
                      c*chunk_size:(c+1)*chunk_size]
            tot = blk.size
            cnt_wall     = np.sum(blk ==   0)
            cnt_non_navi = np.sum(blk <= 230)

            if cnt_wall     / tot >= wall_threshold_ratio:
                grid[r, c] = 2
            elif cnt_non_navi / tot >  non_navigable_threshold_ratio:
                grid[r, c] = 0
            else:
                grid[r, c] = 1

    return grid

if __name__ == '__main__':
    yaml_path = os.path.expanduser('~/grid/sample1.yaml')
    pgm_path  = os.path.expanduser('~/grid/sample1.pgm')
    if not (os.path.exists(yaml_path) and os.path.exists(pgm_path)):
        print("YAML 또는 PGM 파일 경로를 확인하세요."); exit(1)

    chunk_size = 10
    threshold  = 0.3
    coarse = map_to_coarse_grid(yaml_path, pgm_path,
                                chunk_size, threshold)

    # --- 1) 0만 있는 바깥 여백 잘라내기 ---
    coords = np.argwhere(coarse != 0)
    if coords.size == 0:
        print("주행 가능(1) 또는 벽(2) 영역이 없습니다."); exit(0)
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)

    # --- 2) 터미널용 매핑 정의 ---
    # 심플하게 문자로:
    SYMBOL = {
        0: '  ',   # 공백
        1: '..',   # 주행 가능(흰색 영역)
        2: '#',   # 벽(검정 영역)
    }

    # 혹은 ANSI 배경색 블록(지원 터미널 한정):
    ANSI = {
        0: '\033[47m  \033[0m',  # 회색 배경
        1: '\033[107m  \033[0m', # 흰색 배경
        2: '\033[40m  \033[0m',  # 검정 배경
    }

    use_ansi = False  # True로 바꾸면 ANSI 모드

    # --- 3) 출력 ---
    for r in range(rmin, rmax+1):
        row = coarse[r, cmin:cmax+1]
        if use_ansi:
            line = ''.join(ANSI[v] for v in row)
        else:
            line = ''.join(SYMBOL[v] for v in row)
        print(line)
