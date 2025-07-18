#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import os
import rospkg
import cv2
import yaml
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from collections import deque
from algorithms import dijkstra, astar, DStarLite
import threading
from pgm_to_grid import image_Grid

# 1) 목표 이름→월드좌표 (goals.yaml 에서 load됨)
TASK_POSITIONS = {}

# 2) 로봇 목록 (solo)
ROBOTS = ['robot_1']

# 3) 각 로봇별 경로 저장
PATHS = {rb: {'full_path': [], 'current_index': 0, 'active': False} for rb in ROBOTS}

# 4) 마지막 로봇 위치 트래킹 (동적 장애물 업데이트)
LAST_POSITIONS = {rb: None for rb in ROBOTS}

# 5) 퍼블리셔 저장
PUBLISHERS = {}

# 6) 전역 그리드 및 메타데이터
GRID_ORIGIN_X = 0.0
GRID_ORIGIN_Y = 0.0
GRID_CELL_SIZE = 1.0
grid = None
path_lock = threading.Lock()


def load_grid():
    global GRID_ORIGIN_X, GRID_ORIGIN_Y, GRID_CELL_SIZE, grid
    # launch에서 전달된 맵 중심 좌표
    mcx = rospy.get_param('~map_center_x', None)
    mcy = rospy.get_param('~map_center_y', None)
    # pgm, yaml 경로 파라미터
    pgm = rospy.get_param('~pgm_path', None)
    yaml_f = rospy.get_param('~map_yaml', None)
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
        # 셀 크기 계산
        rw = rospy.get_param('~robot_width', 0.55)
        sm = rospy.get_param('~safety_margin', 0.05)
        thresh = rospy.get_param('~wall_thresh', 250)
        cell_m = rw/2 + sm
        pix_per_cell = cell_m / res
        h = max(1, int(img.shape[0] / pix_per_cell))
        w = max(1, int(img.shape[1] / pix_per_cell))
        rospy.loginfo(f"Generating grid from {pgm}: {h}x{w} cells (thresh={thresh})")
        grid, _, _ = image_Grid(h, w, img, wall_thresh=thresh)
        GRID_CELL_SIZE = cell_m
        # 기본 origin
        base_x, base_y = origin[0], origin[1]
        # 맵 중심이 지정되었으면, 그리드 전체 크기 기준 중앙으로 조정
        if mcx is not None and mcy is not None:
            grid_w = w * GRID_CELL_SIZE
            grid_h = h * GRID_CELL_SIZE
            GRID_ORIGIN_X = mcx - grid_w / 2.0
            GRID_ORIGIN_Y = mcy - grid_h / 2.0
        else:
            GRID_ORIGIN_X, GRID_ORIGIN_Y = base_x, base_y
    else:
        # fallback: npy 그리드
        rp = rospkg.RosPack().get_path('path_planning')
        default_np = os.path.join(rp, 'maps', 'grid.npy')
        gp = rospy.get_param('~grid_path', default_np)
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
    row = grid.shape[0] - 1 - row
    row = max(0, min(row, grid.shape[0] - 1))
    col = max(0, min(col, grid.shape[1] - 1))
    return row, col


def grid_to_world(r, c):
    row = max(0, min(r, grid.shape[0] - 1))
    col = max(0, min(c, grid.shape[1] - 1))
    inv = grid.shape[0] - 1 - row
    x = GRID_ORIGIN_X + (col + 0.5) * GRID_CELL_SIZE
    y = GRID_ORIGIN_Y + (inv + 0.5) * GRID_CELL_SIZE
    return x, y


def path_to_PathMsg(path, frame_id='map', z=0.0):
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


def find_current_grid(tf_listener, robot_frame='base_link'):
    try:
        tf_listener.waitForTransform('map', robot_frame, rospy.Time(0), rospy.Duration(1.0))
        trans, _ = tf_listener.lookupTransform('map', robot_frame, rospy.Time(0))
        return world_to_grid(trans[0], trans[1])
    except Exception as e:
        rospy.logwarn(f'TF lookup failed for {robot_frame}: {e}')
        return None


def find_nearest_free(pos):
    q = deque([pos])
    visited = set()
    while q:
        r, c = q.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if grid[r, c] == 0:
            return (r, c)
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                q.append((nr, nc))
    return None


def add_obstacle(cell):
    grid[cell] = 1


def remove_obstacle(cell):
    grid[cell] = 0


def goal_callback(msg, args):
    robot_name, tf_listener, pub, robots = args
    task_name = msg.data.strip()
    if task_name not in TASK_POSITIONS:
        rospy.logerr(f'Unknown goal: {task_name}')
        return
    try:
        trans, _ = tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
    except Exception as e:
        rospy.logwarn(f'[{robot_name}] TF lookup failed: {e}')
        return
    start = world_to_grid(trans[0], trans[1])
    if grid[start] != 0:
        rospy.logwarn(f'Start {start} blocked, finding free...')
        start = find_nearest_free(start)
    goal = world_to_grid(*TASK_POSITIONS[task_name])
    if grid[goal] != 0:
        rospy.logwarn(f'Goal {goal} blocked, finding free...')
        goal = find_nearest_free(goal)
    rospy.loginfo(f'[{robot_name}] Planning from {start} to {goal}')
    path_dyn = dijkstra(grid.copy(), start, goal)
    if path_dyn is None:
        rospy.logwarn('Dijkstra failed, trying A*')
        path_dyn = astar(grid.copy(), start, goal)
    if not path_dyn:
        rospy.logerr(f'No path for {robot_name}')
        return
    with path_lock:
        PATHS[robot_name]['full_path'] = path_dyn
        PATHS[robot_name]['current_index'] = 0
        PATHS[robot_name]['active'] = True
        dstar_instances[robot_name] = None


def step_callback(event, tf_listener):
    global LAST_POSITIONS
    with path_lock:
        for rb in ROBOTS:
            prev = LAST_POSITIONS[rb]
            if prev and grid[prev] == 1:
                remove_obstacle(prev)
            pos = find_current_grid(tf_listener)
            if pos:
                add_obstacle(pos)
                LAST_POSITIONS[rb] = pos
        for rb, info in PATHS.items():
            if not info['active']:
                continue
            full = info['full_path']
            idx = info['current_index']
            if idx >= len(full):
                info['active'] = False
                continue
            remaining = full[idx:]
            msg = path_to_PathMsg(remaining)
            PUBLISHERS[rb].publish(msg)
            info['current_index'] += 1

if __name__ == '__main__':
    rospy.init_node('solo_path_planner_node')
    # goals 및 homes 로드
    if rospy.has_param('goals'):
        TASK_POSITIONS.update({k: tuple(v) for k,v in rospy.get_param('goals').items()})
    if rospy.has_param('homes'):
        TASK_POSITIONS.update({k: tuple(v) for k,v in rospy.get_param('homes').items()})
    rospy.loginfo(f"Goals loaded: {TASK_POSITIONS}")

    # 그리드 로드 및 초기화
    load_grid()

    # D* Lite 인스턴스 초기화
    dstar_instances = {rb: None for rb in ROBOTS}

    # TF listener 설정
    tf_listener = tf.TransformListener()

    # 퍼블리셔/서브스크라이버 설정
    for rb in ROBOTS:
        PUBLISHERS[rb] = rospy.Publisher(f"/{rb}/path", Path, queue_size=10)
        rospy.Subscriber(f"/{rb}/goal", String, goal_callback, (rb, tf_listener, PUBLISHERS[rb], ROBOTS))

    # 주기적 스텝 콜백
    rospy.Timer(rospy.Duration(1.0), lambda ev: step_callback(ev, tf_listener))

    rospy.loginfo("[solo_path_planner_node] Ready (centered grid)!")
    rospy.spin()

