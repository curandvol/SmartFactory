# src/path_planning/grid_utils.py
import os, cv2, yaml, numpy as np
import rospy, rospkg
from pgm_to_grid import image_Grid

GRID_ORIGIN_X = 0.0
GRID_ORIGIN_Y = 0.0
GRID_CELL_SIZE = 1.0

# 그리드 로드
def load_grid():
    """
    동적 모드: launch 파라미터 pgm_path, map_yaml 활용
    fallback: 기존 grid.npy 로드
    """
    pgm = rospy.get_param("~pgm_path", None)
    yaml_f = rospy.get_param("~map_yaml", None)
    if pgm and yaml_f:
        img = cv2.imread(pgm, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rospy.logerr(f"Cannot load PGM: {pgm}")
            rospy.signal_shutdown("Missing map image")
            return
        with open(yaml_f, 'r') as f:
            info = yaml.safe_load(f)
        res = info['resolution']
        origin = info.get('origin', [0.0, 0.0, 0.0])
        rw = rospy.get_param("~robot_width", 0.55)
        sm = rospy.get_param("~safety_margin", 0.05)
        thresh = rospy.get_param("~wall_thresh", 210)
        # 셀 크기 및 그리드 크기
        cell_m = rw/2 + sm
        pix_per_cell = cell_m / res
        h = max(1, int(img.shape[0] / pix_per_cell))
        w = max(1, int(img.shape[1] / pix_per_cell))
        rospy.loginfo(f"Generating grid from {pgm}: {h}x{w} cells (thresh={thresh})")
        grid, ch, cw = image_Grid(h, w, img, wall_thresh=thresh)
        rospy.loginfo(f"  -> grid shape={grid.shape}, cell px={ch}x{cw}")
        # 글로벌 origin/셀 갱신
        global GRID_ORIGIN_X, GRID_ORIGIN_Y, GRID_CELL_SIZE
        GRID_ORIGIN_X = origin[0]
        GRID_ORIGIN_Y = origin[1]
        GRID_CELL_SIZE = cell_m
        return grid
    # fallback: npy 로드
    rp = rospkg.RosPack()
    pkg = rp.get_path('path_planning')
    default_np = os.path.join(pkg, 'maps', 'grid.npy')
    gp = rospy.get_param("~grid_path", default_np)
    while not os.path.exists(gp):
        rospy.logwarn(f"Waiting for grid at {gp}...")
        rospy.sleep(0.5)
    grid = np.load(gp)
    rospy.loginfo(f"Loaded grid from .npy: {grid.shape}")
    return grid

# 월드->그리드 변환
def world_to_grid(x, y):
    col = int((x - GRID_ORIGIN_X) / GRID_CELL_SIZE)
    row = int((y - GRID_ORIGIN_Y) / GRID_CELL_SIZE)
    row = grid.shape[0] - 1 - row
    row = max(0, min(row, grid.shape[0]-1))
    col = max(0, min(col, grid.shape[1]-1))
    return row, col

# 그리드->월드 변환
def grid_to_world(r, c):
    inv = grid.shape[0] - 1 - r
    x = GRID_ORIGIN_X + (c + 0.5) * GRID_CELL_SIZE
    y = GRID_ORIGIN_Y + (inv + 0.5) * GRID_CELL_SIZE
    return x, y
