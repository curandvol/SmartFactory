#!/usr/bin/env python3
import cv2
import numpy as np
import heapq
import time
from pgm_to_grid import image_Grid  # 사용자 정의 격자화 함수


def dijkstra(grid: np.ndarray, start: tuple, end: tuple):
    """
    다익스트라 최단 경로 탐색
    grid: 2D numpy array, 0=free, 1=wall
    start, end: (row, col)
    return: [(r,c), …] 최단 경로 혹은 None
    """
    H, W = grid.shape
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    inf = float('inf')

    dist = [[inf]*W for _ in range(H)]
    prev = [[None]*W for _ in range(H)]
    dist[start[0]][start[1]] = 0
    pq = [(0, start)]

    while pq:
        cost, (r, c) = heapq.heappop(pq)
        if cost > dist[r][c]:
            continue
        if (r, c) == end:
            break
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                ncost = cost + 1
                if ncost < dist[nr][nc]:
                    dist[nr][nc] = ncost
                    prev[nr][nc] = (r, c)
                    heapq.heappush(pq, (ncost, (nr, nc)))

    if dist[end[0]][end[1]] == inf:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur[0]][cur[1]]
    path.reverse()
    return path


def find_alternate_goal(grid: np.ndarray, goal: tuple):
    """
    목표점이 막혔을 때, 인접한 free 셀을 찾아 임시 목표로 반환
    """
    H, W = grid.shape
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    for dr, dc in moves:
        nr, nc = goal[0] + dr, goal[1] + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
            return (nr, nc)
    return goal


def print_subgrid(disp: np.ndarray, r0: int, r1: int, c0: int, c1: int):
    """
    관심 sub-grid 부분만 출력
    """
    print(f"Sub-grid rows {r0}–{r1}, cols {c0}–{c1}:")
    sub = disp[r0:r1+1, c0:c1+1]
    for row in sub:
        print(' '.join(str(int(v)) for v in row))
    print()


def main():
    # 1) 맵 로드 및 격자화
    PGM_FILE = '/home/seongunkim/grid/sample1.pgm'
    GRID_H, GRID_W = 50, 50
    img = cv2.imread(PGM_FILE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load map: {PGM_FILE}")
    grid_list, cell_h, cell_w = image_Grid(GRID_H, GRID_W, img, wall_thresh=210)
    grid = np.array(grid_list, dtype=np.int8)

    # 2) 로봇1/2 시작 및 목표 좌표
    start1, end1 = (30, 17), (25, 34)
    start2, end2 = end1, start1

    # 3) 관심 sub-grid 범위
    r0, r1 = 25, 30
    c0, c1 = 17, 34

    # 4) 초기 위치 및 시간
    pos1, pos2 = start1, start2
    t = 0

    # 초기 상태 출력
    print(f"Time {t}: R1 at {pos1}, R2 at {pos2}")
    disp = grid.copy()
    disp[pos1] = 2
    disp[pos2] = 3
    print_subgrid(disp, r0, r1, c0, c1)

    # 5) 동적 재계획 및 이동 시뮬레이션
    while pos1 != end1 or pos2 != end2:
        t += 1

        # (1) R1 계획: R2 현재 위치를 장애물로 처리
        g1 = grid.copy()
        if pos2 != end1:
            g1[pos2] = 1
        target1 = end1 if g1[end1] == 0 else find_alternate_goal(g1, end1)
        if target1 != end1:
            print(f"R1 목표 {end1}가 막혀, 임시 목표 {target1}로 탐색")
        path1 = dijkstra(g1, pos1, target1)
        next1 = path1[1] if path1 and len(path1) > 1 else pos1

        # (2) R2 계획: R1 위치 및 R1 다음 위치를 장애물로 처리 (우선순위)
        g2 = grid.copy()
        if pos1 != end2:
            g2[pos1] = 1
        if next1 != end2:
            g2[next1] = 1
        target2 = end2 if g2[end2] == 0 else find_alternate_goal(g2, end2)
        if target2 != end2:
            print(f"R2 목표 {end2}가 막혀, 임시 목표 {target2}로 탐색")
        path2 = dijkstra(g2, pos2, target2)
        next2 = path2[1] if path2 and len(path2) > 1 else pos2

        # 위치 업데이트
        pos1, pos2 = next1, next2

        # 표시 그리드: 현재 위치만 2/3 표시
        disp = grid.copy()
        disp[pos1] = 2
        disp[pos2] = 3

        # 출력
        print(f"Time {t}: R1 at {pos1}, R2 at {pos2}")
        print_subgrid(disp, r0, r1, c0, c1)

        # 1초 대기
        time.sleep(1)

    print("Simulation finished.")

if __name__ == '__main__':
    main()

