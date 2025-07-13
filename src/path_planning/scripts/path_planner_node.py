#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import os
import rospkg
import cv2
import yaml
import math
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from collections import deque
from algorithms import dijkstra, heuristic, astar
import threading
from pgm_to_grid import image_Grid

# 1) 목표 이름→월드좌표 (goals.yaml 에서 load됨)
TASK_POSITIONS = {}

# 2) R1/R2 경로 저장
PATHS = {
    'robot_1': {'full_path': [], 'current_index': 0, 'active': False},
    'robot_2': {'full_path': [], 'current_index': 0, 'active': False}
}
LAST_POSITIONS = {'robot_1': None, 'robot_2': None}

# 3) 퍼블리셔 저장
PUBLISHERS = {}

# 4) 글로벌 origin 및 셀 크기
GRID_ORIGIN_X = 0.0
GRID_ORIGIN_Y = 0.0
GRID_CELL_SIZE = 1.0

# 5) 경로 업데이트
PATH_UPDATE_RATE = 1.0
path_lock = threading.Lock()

# 6) 동시 경로 수집
GOAL_COLLECTION = {
    'robot_1': None,
    'robot_2': None,
    'collection_timeout': 2.0,
    'collection_start_time': None,
    'collection_active': False
}
goal_collection_lock = threading.Lock()

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

def world_to_grid(x, y):
    col = (x - GRID_ORIGIN_X) / GRID_CELL_SIZE
    row = (y - GRID_ORIGIN_Y) / GRID_CELL_SIZE
    col = int(math.floor(col))
    row = int(math.floor(row))
    # 2) NumPy 인덱스(row=0이 최상단)이므로 아래로 뒤집기
    row = grid.shape[0] - 1 - row

    # 3) 클램핑
    row = max(0, min(row, grid.shape[0]-1))
    col = max(0, min(col, grid.shape[1]-1))
    return row, col

def grid_to_world(r, c):
    # 3) 클램핑
    row = max(0, min(r, grid.shape[0]-1))
    col = max(0, min(c, grid.shape[1]-1))
    
    inv = grid.shape[0] - 1 - row
    x = GRID_ORIGIN_X + (col + 0.5) * GRID_CELL_SIZE
    y = GRID_ORIGIN_Y + (inv + 0.5) * GRID_CELL_SIZE
    
    return x, y

def path_to_PathMsg(path, frame_id='map',z =0.0):
    msg = Path()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    for r, c in path:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x, pose.pose.position.y = grid_to_world(r, c)
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg

def find_current_grid(tf_listener, robot_frame):
    try:
        now = rospy.Time(0)
        tf_listener.waitForTransform("map", robot_frame, now, rospy.Duration(1.0))
        (trans, _) = tf_listener.lookupTransform("map", robot_frame, now)
        return world_to_grid(trans[0], trans[1])
    except Exception as e:
        rospy.logwarn(f"TF lookup failed for {robot_frame}: {e}")
        return None

def find_nearest_free(grid, pos):
    q = deque([pos])
    visited = set()
    while q:
        r, c = q.popleft()
        if (r, c) in visited or not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            continue
        visited.add((r, c))
        if grid[r][c] == 0:
            return (r, c)
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            q.append((r + dr, c + dc))
    return None
    
def goal_callback(msg, args):
    robot_name, tf_listener, grid, pub, robots = args
    task_name = msg.data.strip()
    if task_name not in TASK_POSITIONS:
        rospy.logerr(f"Unknown goal: {task_name}")
        return

    goal = world_to_grid(*TASK_POSITIONS[task_name])
    start = find_current_grid(tf_listener, f"{robot_name}/base_link")

    if not start:
        rospy.logerr(f"Invalid start position for {robot_name}: {start}")
        return

    if grid[start[0]][start[1]] != 0:
        rospy.logwarn(f"Start at {start} blocked. Searching nearest free...")
        start = find_nearest_free(grid, start)
        if not start:
            rospy.logerr(f"No free start found for {robot_name}")
            return
        rospy.loginfo(f"New start: {start}")

    if grid[goal[0]][goal[1]] != 0:
        rospy.logwarn(f"Goal at {goal} blocked. Searching nearest free...")
        goal = find_nearest_free(grid, goal)
        if not goal:
            rospy.logerr(f"No free goal found for {robot_name}")
            return
        rospy.loginfo(f"New goal: {goal}")

    rospy.loginfo(f"[{robot_name}] Start grid: {start}, Goal grid: {goal}")

    grid_dyn = grid.copy()
    for other in robots:
        if other == robot_name:
            continue
        pos = find_current_grid(tf_listener, f"{other}/base_link")
        if pos and 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]:
            grid_dyn[pos[0], pos[1]] = 1

    # 1) Dijkstra로 전체 경로 계산
    path = dijkstra(grid_dyn, start, goal)
    if path is None:
        rospy.logerr(f"[{robot_name}] No path found by Dijkstra from {start} to {goal}")
        return

    # 2) Robot 2 우선순위: robot_1의 머리 부분만 블록 처리하여 A* 재계산
    if robot_name == 'robot_2' and PATHS['robot_1']['full_path']:
        head = PATHS['robot_1']['current_index']
        lookahead = 10  # 앞으로 이만큼 칸만 블록 처리
        block_cells = PATHS['robot_1']['full_path'][head:head + lookahead]
        overlap = set(block_cells) & set(path)
        if overlap:
            rospy.logwarn(f"[{robot_name}] Overlap with robot_1 ahead ({len(overlap)} cells), replanning with A*")
            grid_block = grid_dyn.copy()
            for (r, c) in block_cells:
                grid_block[r, c] = 1
            new_path = astar(grid_block, start, goal)
            if new_path is None:
                rospy.logerr(f"[{robot_name}] A* replanning failed")
                return
            path = new_path
            rospy.loginfo(f"[{robot_name}] A* replanned path ({len(path)} cells) ready")

    # 3) 최종 경로 저장 및 활성화
    with path_lock:
        PATHS[robot_name]['full_path']    = path
        PATHS[robot_name]['current_index'] = 0
        PATHS[robot_name]['active']       = True
    rospy.loginfo(f"[{robot_name}] final path ready ({len(path)} cells), starting 1 Hz stepping")

if __name__ == '__main__':
    rospy.init_node('path_planner_node')
    TASK_POSITIONS.clear()
    if rospy.has_param('goals'):
        raw = rospy.get_param('goals')
        TASK_POSITIONS.update({k: tuple(v) for k, v in raw.items()})
    if rospy.has_param('homes'):
        raw_homes = rospy.get_param('homes')
        TASK_POSITIONS.update({k: tuple(v) for k, v in raw_homes.items()})
    rospy.loginfo(f"Goals loaded: {TASK_POSITIONS}")
        
    # 그리드 로드
    grid = load_grid()
    map_msg = rospy.wait_for_message('/map', OccupancyGrid)
    down = rospy.get_param('~downsample_factor', 1)
    rospy.loginfo(f"Using map origin=({GRID_ORIGIN_X:.3f}, {GRID_ORIGIN_Y:.3f}), "
                  f"cell_size={GRID_CELL_SIZE:.3f}")
    
    #GRID_CELL_SIZE = map_msg.info.resolution * down
    GRID_ORIGIN_X  = map_msg.info.origin.position.x
    GRID_ORIGIN_Y  = map_msg.info.origin.position.y
    
    # TF 리스너 & 퍼블리셔/구독자 설정
    tf_listener = tf.TransformListener()
    robots = ['robot_1', 'robot_2']
    for robot in robots:
        PUBLISHERS[robot] = rospy.Publisher(f"/{robot}/path", Path, queue_size=10)
        rospy.Subscriber(f"/{robot}/goal", String, goal_callback,
                         (robot, tf_listener, grid, PUBLISHERS[robot], robots))

    # 1 Hz 스텝 퍼블리시 타이머
    def step_callback(event):
        with path_lock:
            for rb, info in PATHS.items():
                if not info['active']:
                    continue
                idx      = info['current_index']
                full     = info['full_path']
                remaining = full[idx:]
                if not remaining:
                    info['active'] = False
                    rospy.loginfo(f"[{rb}] reached goal, stopping")
                    continue
                    
                z= 1.0 if rb == 'robot_2' else 0.0
                msg = path_to_PathMsg(remaining,z=z)
                PUBLISHERS[rb].publish(msg)
                info['current_index'] += 1

    rospy.Timer(rospy.Duration(1.0), step_callback)

    rospy.loginfo("[path_planner_node] Ready for goal commands.")
    rospy.spin()

