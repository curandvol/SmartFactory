# src/path_planning/grid_utils.py
import os, cv2, yaml, numpy as np
import rospy, rospkg
from pgm_to_grid import image_Grid

# 전역 변수들
grid = None
GRID_ORIGIN_X = 0.0
GRID_ORIGIN_Y = 0.0
GRID_CELL_SIZE = 1.0

def load_grid():
    """
    동적 모드: launch 파라미터 pgm_path, map_yaml 활용
    fallback: 기존 grid.npy 로드
    """
    global grid, GRID_ORIGIN_X, GRID_ORIGIN_Y, GRID_CELL_SIZE
    
    pgm = rospy.get_param("~pgm_path", None)
    yaml_f = rospy.get_param("~map_yaml", None)
    
    if pgm and yaml_f:
        # PGM 파일로부터 그리드 생성
        img = cv2.imread(pgm, cv2.IMREAD_GRAYSCALE)
        if img is None:
            rospy.logerr(f"Cannot load PGM: {pgm}")
            rospy.signal_shutdown("Missing map image")
            return None
            
        with open(yaml_f, 'r') as f:
            info = yaml.safe_load(f)
        
        res = info['resolution']
        origin = info.get('origin', [0.0, 0.0, 0.0])
        
        rw = rospy.get_param("~robot_width", 0.55)
        sm = rospy.get_param("~safety_margin", 0.05)
        thresh = rospy.get_param("~wall_thresh", 210)
        
        # 셀 크기 및 그리드 크기 계산
        cell_m = rw/2 + sm
        pix_per_cell = cell_m / res
        
        h = max(1, int(img.shape[0] / pix_per_cell))
        w = max(1, int(img.shape[1] / pix_per_cell))
        
        rospy.loginfo(f"Generating grid from {pgm}: {h}x{w} cells (thresh={thresh})")
        grid, ch, cw = image_Grid(h, w, img, wall_thresh=thresh)
        rospy.loginfo(f"Grid generation complete: shape={grid.shape}, cell px={ch}x{cw}")
        
        # 전역 변수 설정
        GRID_ORIGIN_X, GRID_ORIGIN_Y = origin[0], origin[1]
        GRID_CELL_SIZE = cell_m
        
        rospy.loginfo(f"Grid parameters: origin=({GRID_ORIGIN_X:.2f}, {GRID_ORIGIN_Y:.2f}), "
                     f"cell_size={GRID_CELL_SIZE:.3f}m")
        return grid
    
    else:
        # Fallback: npy 파일 로드
        rp = rospkg.RosPack()
        pkg = rp.get_path('path_planning')
        default_np = os.path.join(pkg, 'maps', 'grid.npy')
        gp = rospy.get_param("~grid_path", default_np)
        
        while not os.path.exists(gp):
            rospy.logwarn(f"Waiting for grid at {gp}...")
            rospy.sleep(0.5)
        
        grid = np.load(gp)
        rospy.loginfo(f"Loaded grid from .npy: {grid.shape}")
        
        # 기본값 설정 (필요시 파라미터로 조정)
        GRID_ORIGIN_X = rospy.get_param("~grid_origin_x", 0.0)
        GRID_ORIGIN_Y = rospy.get_param("~grid_origin_y", 0.0)
        GRID_CELL_SIZE = rospy.get_param("~grid_cell_size", 1.0)
        
        return grid

def world_to_grid(x, y):
    """월드 좌표를 그리드 좌표로 변환"""
    global grid, GRID_ORIGIN_X, GRID_ORIGIN_Y, GRID_CELL_SIZE
    
    if grid is None:
        rospy.logerr("Grid not loaded!")
        return None
    
    # 월드 좌표를 그리드 인덱스로 변환
    col = int((x - GRID_ORIGIN_X) / GRID_CELL_SIZE)
    row = int((y - GRID_ORIGIN_Y) / GRID_CELL_SIZE)
    
    # Y축 뒤집기 (OpenCV 이미지 좌표계 -> 일반 좌표계)
    row = grid.shape[0] - 1 - row
    
    # 경계 검사 및 클램핑
    row = max(0, min(row, grid.shape[0] - 1))
    col = max(0, min(col, grid.shape[1] - 1))
    
    # 디버그 로그 (필요시 주석 해제)
    # rospy.logdebug(f"world_to_grid: ({x:.2f}, {y:.2f}) -> ({row}, {col})")
    
    return row, col

def grid_to_world(r, c):
    """그리드 좌표를 월드 좌표로 변환"""
    global grid, GRID_ORIGIN_X, GRID_ORIGIN_Y, GRID_CELL_SIZE
    
    if grid is None:
        rospy.logerr("Grid not loaded!")
        return None, None
    
    # 경계 검사 및 클램핑
    r = max(0, min(r, grid.shape[0] - 1))
    c = max(0, min(c, grid.shape[1] - 1))
    
    # Y축 뒤집기 (일반 좌표계 -> OpenCV 이미지 좌표계)
    row = grid.shape[0] - 1 - r
    
    # 그리드 인덱스를 월드 좌표로 변환 (셀 중심점)
    x = GRID_ORIGIN_X + (c + 0.5) * GRID_CELL_SIZE
    y = GRID_ORIGIN_Y + (row + 0.5) * GRID_CELL_SIZE
    
    # 디버그 로그 (상세한 변환 과정 표시)
    rospy.logdebug(f"grid_to_world: Grid({r}, {c}) -> World({x:.2f}, {y:.2f})")
    
    return x, y

def is_valid_grid_position(r, c):
    """그리드 위치가 유효한지 확인"""
    global grid
    
    if grid is None:
        return False
    
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]

def is_free_cell(r, c):
    """그리드 셀이 자유 공간인지 확인"""
    global grid
    
    if not is_valid_grid_position(r, c):
        return False
    
    return grid[r, c] == 0

def get_grid_info():
    """그리드 정보 반환"""
    global grid, GRID_ORIGIN_X, GRID_ORIGIN_Y, GRID_CELL_SIZE
    
    if grid is None:
        return None
    
    return {
        'shape': grid.shape,
        'origin_x': GRID_ORIGIN_X,
        'origin_y': GRID_ORIGIN_Y,
        'cell_size': GRID_CELL_SIZE,
        'width': grid.shape[1],
        'height': grid.shape[0]
    }

def print_grid_info():
    """그리드 정보 출력"""
    info = get_grid_info()
    if info:
        rospy.loginfo(f"Grid Info:")
        rospy.loginfo(f"  Shape: {info['shape']}")
        rospy.loginfo(f"  Origin: ({info['origin_x']:.2f}, {info['origin_y']:.2f})")
        rospy.loginfo(f"  Cell Size: {info['cell_size']:.3f}m")
        rospy.loginfo(f"  Dimensions: {info['width']} x {info['height']}")
    else:
        rospy.logwarn("Grid not loaded!")
