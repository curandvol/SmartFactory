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
from algorithms import dijkstra, heuristic, astar, DStarLite
import threading
from pgm_to_grid import image_Grid

# 1) 목표 이름→월드좌표 (goals.yaml 에서 load됨)
TASK_POSITIONS = {}

# 2) R1/R2 경로 저장
PATHS = {
    'robot_1': {'full_path': [], 'current_index': 0, 'active': False},
    'robot_2': {'full_path': [], 'current_index': 0, 'active': False}
}

# 3) 마지막 로봇 위치 트래킹 (동적 로봇 장애물 업데이트 용)
LAST_POSITIONS = {'robot_1': None, 'robot_2': None}

# 4) 퍼블리셔 저장
PUBLISHERS = {}

# 5) 글로벌 origin 및 셀 크기
GRID_ORIGIN_X = 0.0
GRID_ORIGIN_Y = 0.0
GRID_CELL_SIZE = 1.0

grid = None  # 전역 그리드
path_lock = threading.Lock()

def load_grid():
    global GRID_ORIGIN_X, GRID_ORIGIN_Y, GRID_CELL_SIZE, grid
    # 동적 모드: launch 파라미터 pgm_path, map_yaml 활용
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
        thresh = rospy.get_param("~wall_thresh", 250)
        cell_m = rw/2 + sm
        pix_per_cell = cell_m / res
        h = max(1, int(img.shape[0] / pix_per_cell))
        w = max(1, int(img.shape[1] / pix_per_cell))
        rospy.loginfo(f"Generating grid from {pgm}: {h}x{w} cells (thresh={thresh})")
        grid, ch, cw = image_Grid(h, w, img, wall_thresh=thresh)
        GRID_ORIGIN_X, GRID_ORIGIN_Y = origin[0], origin[1]
        GRID_CELL_SIZE = cell_m
    else:
        # fallback: npy 로드
        rp = rospkg.RosPack().get_path('path_planning')
        default_np = os.path.join(rp, 'maps', 'grid.npy')
        gp = rospy.get_param("~grid_path", default_np)
        while not os.path.exists(gp):
            rospy.logwarn(f"Waiting for grid at {gp}...")
            rospy.sleep(0.5)
        grid = np.load(gp)
        rospy.loginfo(f"Loaded grid from .npy: {grid.shape}")
    return grid

# 좌표 변환 함수
def world_to_grid(x, y):
    col = (x - GRID_ORIGIN_X) / GRID_CELL_SIZE
    row = (y - GRID_ORIGIN_Y) / GRID_CELL_SIZE
    col = int(math.floor(col))
    row = int(math.floor(row))
    row = grid.shape[0] - 1 - row
    row = max(0, min(row, grid.shape[0]-1))
    col = max(0, min(col, grid.shape[1]-1))
    return row, col

# 그리드→월드 좌표 변환
def grid_to_world(r, c):
    row = max(0, min(r, grid.shape[0]-1))
    col = max(0, min(c, grid.shape[1]-1))
    inv = grid.shape[0] - 1 - row
    x = GRID_ORIGIN_X + (col + 0.5) * GRID_CELL_SIZE
    y = GRID_ORIGIN_Y + (inv + 0.5) * GRID_CELL_SIZE
    return x, y

# PathMsg 생성
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

# 현재 로봇 그리드 셀 찾기
def find_current_grid(tf_listener, robot_frame):
    try:
        now = rospy.Time(0)
        tf_listener.waitForTransform("map", robot_frame, now, rospy.Duration(1.0))
        (trans, _) = tf_listener.lookupTransform("map", robot_frame, now)
        return world_to_grid(trans[0], trans[1])
    except Exception as e:
        rospy.logwarn(f"TF lookup failed for {robot_frame}: {e}")
        return None

# 가장 가까운 빈 칸 찾기
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

# 로봇 장애물 추가/제거
def add_obstacle(cell):
    global grid
    grid[cell] = 1
    #튜플을 인덱스로 바로 사용 (grid[r,c] = 1)
def remove_obstacle(cell):
    global grid
    grid[cell] = 0

# 목적지 콜백: 전역 grid 사용
def goal_callback(msg, args):
    robot_name, tf_listener, pub, robots = args
    task_name = msg.data.strip()
    if task_name not in TASK_POSITIONS:
        rospy.logerr(f"Unknown goal: {task_name}")
        return
    # 매 초마다 로봇의 위치를 저장하고, 현재 로봇의 위치를 시작점으로 계산해 도착지점을 계산
    # 1) TF에서 로봇 현재 위치(x,y) 읽어오기
    try:
        trans, _ = tf_listener.lookupTransform("map",
                                               f"{robot_name}/base_link",
                                               rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn(f"[{robot_name}] TF lookup failed: {e}")
        return

    # 2) world → grid 변환
    start = world_to_grid(trans[0], trans[1])

    # 3) 디버그 출력 (world 좌표, grid 인덱스, 그리드 값)
    rospy.loginfo(f"[{robot_name}] Start world=({trans[0]:.2f},{trans[1]:.2f}) → "
                  f"grid=({start[0]},{start[1]}), value={grid[start]}")
    if not start:
        rospy.logerr(f"Invalid Start for {robot_name} : {start}")
        return
    LAST_POSITIONS[robot_name] = start
    goal = world_to_grid(*TASK_POSITIONS[task_name])
    
    if not start:
        rospy.logerr(f"Invalid start for {robot_name}: {start}")
        return
    if grid[start] != 0:
        rospy.logwarn(f"Start {start} blocked, finding free...")
        start = find_nearest_free(start)
    if grid[goal] != 0:
        rospy.logwarn(f"Goal {goal} blocked, finding free...")
        goal = find_nearest_free(goal)
    rospy.loginfo(f"[{robot_name}] Planning from {start} to {goal}")
    path_dyn = dijkstra(grid.copy(), start, goal) #전역 경로 생성
    if path_dyn is None: #다익스트라 생성에 실패했을 때 -> 전역 경로 생성 = A*
        rospy.logwarn(f"Dijkstra failed, trying A*")
        path_dyn = astar(grid.copy(), start, goal)
    if not path_dyn:
        rospy.logerr(f"No path for {robot_name}") #A*가 실패했을때 -> 경로 없음 판정
        return
    with path_lock:
        PATHS[robot_name]['full_path'] = path_dyn
        PATHS[robot_name]['current_index'] = 0
        PATHS[robot_name]['active'] = True
        dstar_instances[robot_name] = None

def step_callback(event):
    global LAST_POSITIONS
    with path_lock:
        # 동적 로봇 위치로 장애물 업데이트
        for rb in PATHS:
            prev = LAST_POSITIONS[rb]
            if prev and grid[prev] == 1:
                remove_obstacle(prev)
            pos = find_current_grid(tf_listener, f"{rb}/base_link")
            if pos:
                add_obstacle(pos)
                LAST_POSITIONS[rb] = pos
        # 각 로봇별로 경로 진행 및 필요 시 D* Lite 재계획
        for rb, info in PATHS.items():
            if not info['active']:
                continue
            full = info['full_path']
            idx = info['current_index']
            if idx >= len(full):
                info['active'] = False
                continue
            current = full[idx]
            goal = full[-1]
            next_cell = current
            # 장애물 발견 시 증분 D* Lite 재계획
            if grid[next_cell] == 1:
                if dstar_instances[rb] is None:
                    dstar_instances[rb] = DStarLite(grid.copy(), current, goal)
                else:
                    dstar_instances[rb].update_obstacles([next_cell])
                new_path = dstar_instances[rb].replan(current, goal)
                if new_path:
                    info['full_path'] = new_path
                    info['current_index'] = 0
                    full = new_path
                    idx = 0
            # 퍼블리시
            remaining = full[idx:]
            msg = path_to_PathMsg(remaining, z=1.0 if rb=='robot_2' else 0.0)
            PUBLISHERS[rb].publish(msg)
            info['current_index'] += 1

if __name__ == '__main__':
    rospy.init_node('path_planner_node')
    # TASK_POSITIONS 로드
    if rospy.has_param('goals'):
        TASK_POSITIONS.update({k: tuple(v) for k, v in rospy.get_param('goals').items()})
    if rospy.has_param('homes'):
        TASK_POSITIONS.update({k: tuple(v) for k, v in rospy.get_param('homes').items()})
    rospy.loginfo(f"Goals: {TASK_POSITIONS}")
    # 그리드 로드
    load_grid()
    # D* Lite 인스턴스 초기화
    dstar_instances = {'robot_1': None, 'robot_2': None}
    # TF listener, publishers, subscribers 설정
    tf_listener = tf.TransformListener()
    for rb in PATHS:
        PUBLISHERS[rb] = rospy.Publisher(f"/{rb}/path", Path, queue_size=10)
        rospy.Subscriber(f"/{rb}/goal", String, goal_callback, (rb, tf_listener, PUBLISHERS[rb], list(PATHS.keys())))
    rospy.Timer(rospy.Duration(1.0), step_callback)
    rospy.loginfo("[path_planner_node] Ready for goal commands.")
    rospy.spin()

