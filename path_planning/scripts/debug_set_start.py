#!/usr/bin/env python3
import numpy as np
import os
import subprocess
import sys

# grid, 맵 파라미터
GRID_PATH = "/home/yujin/smart_factory_ws/src/path_planning/maps/grid.npy"
MAP_RESOLUTION = 0.05
MAP_ORIGIN = (-10.0, -10.0)

def grid_to_world(r, c):
    x = MAP_ORIGIN[0] + (c + 0.5) * MAP_RESOLUTION
    y = MAP_ORIGIN[1] + (r + 0.5) * MAP_RESOLUTION
    return x, y

def main():
    if len(sys.argv) < 2:
        print("[ERROR] Usage: rosrun path_planning debug_set_start.py <robot_name>")
        return

    robot = sys.argv[1]

    if not os.path.exists(GRID_PATH):
        print(f"[ERROR] Grid file not found: {GRID_PATH}")
        return

    grid = np.load(GRID_PATH)

    # free cell 찾기
    found = False
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r][c] == 0:
                x, y = grid_to_world(r, c)
                print(f"[INFO] Found free cell at grid ({r},{c}), world ({x:.2f},{y:.2f})")

                # static_transform_publisher 실행
                cmd = [
                    "rosrun", "tf", "static_transform_publisher",
                    f"{x}", f"{y}", "0",
                    "0", "0", "0",
                    "map", f"{robot}/odom", "100"
                ]
                print(f"[INFO] Running: {' '.join(cmd)}")
                subprocess.Popen(cmd)
                found = True
                break
        if found:
            break

    if not found:
        print("[ERROR] No free cell found in grid map!")

if __name__ == "__main__":
    main()
