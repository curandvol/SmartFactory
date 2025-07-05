#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import heapq
import os
import rospkg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from collections import deque
import threading
import time

# 1) 목표 이름→월드좌표 (goals.yaml 로드 시 덮어쓰기)
TASK_POSITIONS = {
    'goal1': (-3.0, -1.5),
    'goal2': (-2.5, -1.0)
}

# 2) R1/R2 경로 저장 (전체 경로와 현재 인덱스)
PATHS = {
    'robot_1': {'full_path': [], 'current_index': 0, 'active': False},
    'robot_2': {'full_path': [], 'current_index': 0, 'active': False}
}

# 3) 맵 원점·해상도 (파라미터로 재설정됨)
MAP_ORIGIN = (-3.666302, -1.963480)
MAP_RESOLUTION = 0.05

# 4) 경로 업데이트 타이머
PATH_UPDATE_RATE = 1.0  # 1초마다 한 칸씩 이동
path_lock = threading.Lock()
 
# 5) 동시 경로 생성을 위한 목표 수집
GOAL_COLLECTION = {
    'robot_1': None,
    'robot_2': None,
    'collection_timeout': 2.0,  # 2초 대기
    'collection_start_time': None,
    'collection_active': False
}
goal_collection_lock = threading.Lock()

# --- Dijkstra ---
def dijkstra(grid, start, end):
    H, W = grid.shape
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            break
        if d > dist[u]:
            continue
        for dr, dc in moves:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < H and 0 <= v[1] < W and grid[v] == 0:
                nd = d + 1
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
    
    if end not in prev and start != end:
        return None
    
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]

# --- A* (for R2 우회 및 초기 실패 시) ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, end):
    H, W = grid.shape
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start, None))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if parent is not None and current not in came_from:
            came_from[current] = parent
        if current == end:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            return path[::-1]
        
        for dr, dc in moves:
            nbr = (current[0] + dr, current[1] + dc)
            if 0 <= nbr[0] < H and 0 <= nbr[1] < W and grid[nbr] == 0:
                tg = g + 1
                if nbr not in g_score or tg < g_score[nbr]:
                    g_score[nbr] = tg
                    heapq.heappush(open_set, (tg + heuristic(nbr, end), tg, nbr, current))
    return None

# 그리드 로드
def load_grid():
    rp = rospkg.RosPack()
    pkg = rp.get_path('path_planning')
    default_np = os.path.join(pkg, 'maps', 'grid.npy')
    gp = rospy.get_param("~grid_path", default_np)
    while not os.path.exists(gp):
        rospy.logwarn(f"Waiting for grid at {gp}...")
        rospy.sleep(0.5)
    g = np.load(gp)
    rospy.loginfo(f"Loaded grid: {g.shape}")
    return g

# 월드→그리드 좌표 변환
def world_to_grid(x, y):
    col = int((x - GRID_ORIGIN_X) / GRID_CELL_SIZE)
    row = int((y - GRID_ORIGIN_Y) / GRID_CELL_SIZE)
    row = grid.shape[0] - 1 - row
    row = max(0, min(row, grid.shape[0] - 1))
    col = max(0, min(col, grid.shape[1] - 1))
    return row, col

# 그리드→월드 좌표 변환
def grid_to_world(r, c):
    inv = grid.shape[0] - 1 - r
    inv = max(0, min(inv, grid.shape[0] - 1))
    c = max(0, min(c, grid.shape[1] - 1))
    x = GRID_ORIGIN_X + (c + 0.5) * GRID_CELL_SIZE
    y = GRID_ORIGIN_Y + (inv + 0.5) * GRID_CELL_SIZE
    return x, y

# Path 메시지 생성
def path_to_msg(p, fid='map'):
    msg = Path()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = fid
    for r, c in p:
        ps = PoseStamped()
        ps.header.frame_id = fid
        ps.pose.position.x, ps.pose.position.y = grid_to_world(r, c)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg

# TF에서 현재 그리드 셀 얻기
def find_current_grid(tf_listener, frame):
    try:
        now = rospy.Time(0)
        tf_listener.waitForTransform('map', frame, now, rospy.Duration(1.0))
        t, _ = tf_listener.lookupTransform('map', frame, now)
        return world_to_grid(t[0], t[1])
    except Exception as e:
        rospy.logwarn(f"TF lookup failed for {frame}: {e}")
        return None

# 가장 가까운 free 셀 탐색
def find_nearest_free(grid, cell):
    H, W = grid.shape
    q = deque([cell])
    vis = {cell}
    moves = [(1,0),(-1,0),(0,1),(-1,0)]
    while q:
        r, c = q.popleft()
        if grid[r, c] == 0:
            return (r, c)
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis:
                vis.add((nr, nc))
                q.append((nr, nc))
    return cell

# 미래 경로 충돌 검사
def check_future_collision(path1, path2, start_time=0):
    """두 경로가 미래에 충돌하는지 확인"""
    if not path1 or not path2:
        return False, -1
    
    max_len = max(len(path1), len(path2))
    for t in range(start_time, max_len):
        cell1 = path1[t] if t < len(path1) else path1[-1]
        cell2 = path2[t] if t < len(path2) else path2[-1]
        
        if cell1 == cell2:
            return True, t
    
    return False, -1

# R2 우회 경로 생성 (A* 사용)
def generate_r2_bypass_path(grid_arr, start, goal, r1_path):
    """R1 경로를 피해서 R2의 우회 경로 생성"""
    g2_bypass = grid_arr.copy()
    for cell in r1_path:
        if cell != start:  # 시작점은 제외
            g2_bypass[cell] = 1
    
    bypass_path = astar(g2_bypass, start, goal)
    
    if not bypass_path:
        rospy.logwarn("R2: A* 우회 경로 생성 실패, 제약 없는 경로 시도")
        free_goal = goal if grid_arr[goal] == 0 else find_nearest_free(grid_arr, goal)
        bypass_path = astar(grid_arr, start, free_goal)
    
    return bypass_path

# 동시 경로 생성 (핵심 로직) - 수정된 버전
def generate_simultaneous_paths(tf_listener, grid_arr, goals):
    """두 로봇의 경로를 동시에 생성"""
    rospy.loginfo("=== 동시 경로 생성 시작 ===")
    
    # 현재 위치 확인
    positions = {}
    for robot_name in goals:
        pos = find_current_grid(tf_listener, f"{robot_name}/base_link")
        if not pos:
            rospy.logerr(f"{robot_name}: 현재 위치 확인 실패")
            return None
        
        # <<<--- [수정된 부분 시작] --- >>>
        # 현재 위치가 장애물인지 확인하고, 장애물이라면 가장 가까운 유효한 셀로 보정
        if grid_arr[pos] == 1:
            rospy.logwarn(f"{robot_name}: 시작 위치 {pos}가 장애물 위에 있습니다. 가장 가까운 유효 지점을 탐색합니다.")
            pos = find_nearest_free(grid_arr, pos)
            rospy.loginfo(f"{robot_name}: 보정된 시작 위치: {pos}")
        # <<<--- [수정된 부분 끝] --- >>>

        positions[robot_name] = pos
    
    # 목표 위치 변환
    goal_positions = {}
    for robot_name, goal_name in goals.items():
        if goal_name not in TASK_POSITIONS:
            rospy.logerr(f"Unknown goal: {goal_name}")
            return None
        goal_pos = world_to_grid(*TASK_POSITIONS[goal_name])
        if grid_arr[goal_pos] == 1:
            goal_pos = find_nearest_free(grid_arr, goal_pos)
        goal_positions[robot_name] = goal_pos
    
    paths = {}
    
    # 1) R1 경로 생성 (우선순위 - Dijkstra)
    if 'robot_1' in goals:
        rospy.loginfo("R1: 우선 경로 생성 (Dijkstra)")
        start = positions['robot_1']
        goal = goal_positions['robot_1']
        
        # R2 현재 위치만 고려
        g1 = grid_arr.copy()
        if 'robot_2' in positions and positions['robot_2'] != start:
            g1[positions['robot_2']] = 1
        
        r1_path = dijkstra(g1, start, goal)
        if not r1_path:
            rospy.logerr("R1: Dijkstra 경로 생성 실패")
            return None
        
        paths['robot_1'] = r1_path
        rospy.loginfo(f"R1: 경로 생성 완료 ({len(r1_path)} 칸)")
    
    # 2) R2 경로 생성 (R1 경로 고려)
    if 'robot_2' in goals:
        rospy.loginfo("R2: 충돌 회피 경로 생성")
        start = positions['robot_2']
        goal = goal_positions['robot_2']
        
        # 초기 경로 (R1 현재 위치만 고려)
        g2 = grid_arr.copy()
        if 'robot_1' in positions and positions['robot_1'] != start:
            g2[positions['robot_1']] = 1
        
        # Dijkstra 시도
        r2_path = dijkstra(g2, start, goal)
        if not r2_path:
            rospy.logwarn("R2: Dijkstra 실패, A* 시도")
            r2_path = astar(g2, start, goal)
        
        if not r2_path:
            rospy.logwarn("R2: 제약 경로 실패, 제약 없는 A* 시도")
            free_goal = goal if grid_arr[goal] == 0 else find_nearest_free(grid_arr, goal)
            r2_path = astar(grid_arr, start, free_goal)
        
        if not r2_path:
            rospy.logerr("R2: 모든 경로 생성 방법 실패")
            return None
        
        # R1 경로와 충돌 검사
        if 'robot_1' in paths:
            r1_path = paths['robot_1']
            has_collision, collision_time = check_future_collision(r1_path, r2_path)
            
            if has_collision:
                rospy.logwarn(f"R2: R1과 충돌 예상 (시간 {collision_time}), 우회 경로 생성")
                bypass_path = generate_r2_bypass_path(grid_arr, start, goal, r1_path)
                if bypass_path:
                    r2_path = bypass_path
                    rospy.loginfo("R2: 우회 경로 생성 성공")
                else:
                    rospy.logwarn("R2: 우회 경로 생성 실패, 기존 경로 사용")
            else:
                rospy.loginfo("R2: R1과 충돌 없음")
        
        paths['robot_2'] = r2_path
        rospy.loginfo(f"R2: 경로 생성 완료 ({len(r2_path)} 칸)")
    
    rospy.loginfo("=== 동시 경로 생성 완료 ===")
    return paths

# 목표 수집 타이머 콜백
def collection_timer_callback():
    """목표 수집 타이머 만료 시 호출"""
    with goal_collection_lock:
        if not GOAL_COLLECTION['collection_active']:
            return
        
        rospy.loginfo("목표 수집 시간 만료, 수집된 목표로 경로 생성")
        collected_goals = {}
        
        for robot_name in ['robot_1', 'robot_2']:
            if GOAL_COLLECTION[robot_name] is not None:
                collected_goals[robot_name] = GOAL_COLLECTION[robot_name]
        
        # 수집된 목표로 경로 생성
        if collected_goals:
            process_collected_goals(collected_goals)
        
        # 수집 상태 리셋
        reset_goal_collection()

def reset_goal_collection():
    """목표 수집 상태 리셋"""
    GOAL_COLLECTION['robot_1'] = None
    GOAL_COLLECTION['robot_2'] = None
    GOAL_COLLECTION['collection_active'] = False
    GOAL_COLLECTION['collection_start_time'] = None

def process_collected_goals(goals):
    """수집된 목표들로 경로 생성 및 발행"""
    global publishers
    
    rospy.loginfo(f"수집된 목표로 경로 생성: {goals}")
    
    # 동시 경로 생성
    paths = generate_simultaneous_paths(tf_listener, grid, goals)
    
    if not paths:
        rospy.logerr("동시 경로 생성 실패")
        return
    
    # 경로 저장 및 발행
    with path_lock:
        for robot_name, path in paths.items():
            PATHS[robot_name] = {
                'full_path': path,
                'current_index': 0,
                'active': True
            }
            
            # 경로 발행
            if robot_name in publishers:
                publishers[robot_name].publish(path_to_msg(path))
            
            rospy.loginfo(f"{robot_name}: 경로 활성화 완료")

# 경로 업데이트 타이머 콜백
def update_paths():
    """매 초마다 호출되어 로봇들의 경로를 한 칸씩 진행"""
    global publishers
    
    with path_lock:
        for robot_name in ['robot_1', 'robot_2']:
            if not PATHS[robot_name]['active']:
                continue
            
            path_data = PATHS[robot_name]
            current_idx = path_data['current_index']
            full_path = path_data['full_path']
            
            # 경로 완료 체크
            if current_idx >= len(full_path) - 1:
                rospy.loginfo(f"{robot_name}: 목표 도달, 경로 완료")
                PATHS[robot_name]['active'] = False
                continue
            
            # 다음 칸으로 이동
            PATHS[robot_name]['current_index'] = current_idx + 1
            
            # 남은 경로 발행 (현재 위치 포함)
            remaining_path = full_path[current_idx + 1:]
            if remaining_path and robot_name in publishers:
                publishers[robot_name].publish(path_to_msg(remaining_path))
            
            rospy.loginfo(f"{robot_name}: 경로 진행 ({current_idx + 1}/{len(full_path)}), 남은 경로: {len(remaining_path)} 칸")

# 목표 콜백 (수정된 버전)
def goal_callback(msg, args):
    """목표 수신 시 수집 모드로 처리"""
    global publishers
    robot, tf_listener, grid_arr, pub, robots = args
    
    goal_name = msg.data.strip()
    rospy.loginfo(f"=== {robot} 목표 수신: {goal_name} ===")
    
    if goal_name not in TASK_POSITIONS:
        rospy.logerr(f"Unknown goal: {goal_name}")
        return
    
    with goal_collection_lock:
        # 목표 수집 시작
        GOAL_COLLECTION[robot] = goal_name
        
        if not GOAL_COLLECTION['collection_active']:
            # 첫 번째 목표 - 수집 모드 시작
            GOAL_COLLECTION['collection_active'] = True
            GOAL_COLLECTION['collection_start_time'] = rospy.Time.now()
            
            # 타이머 설정
            rospy.Timer(rospy.Duration(GOAL_COLLECTION['collection_timeout']), 
                        lambda event: collection_timer_callback(), oneshot=True)
            
            rospy.loginfo(f"목표 수집 시작, {GOAL_COLLECTION['collection_timeout']}초 대기")
        
        # 모든 로봇의 목표가 수집되었는지 확인
        collected_goals = {}
        for robot_name in ['robot_1', 'robot_2']:
            if GOAL_COLLECTION[robot_name] is not None:
                collected_goals[robot_name] = GOAL_COLLECTION[robot_name]
        
        # 모든 로봇의 목표가 수집되면 즉시 처리
        if len(collected_goals) == 2:
            rospy.loginfo("모든 로봇의 목표 수집 완료, 즉시 경로 생성")
            process_collected_goals(collected_goals)
            reset_goal_collection()

# 전역 변수
publishers = {}

if __name__ == '__main__':
    rospy.init_node('path_planner_node')

    # 목표 위치 로드
    if rospy.has_param('goals'):
        raw = rospy.get_param('goals')
        TASK_POSITIONS.clear()
        TASK_POSITIONS.update({k: tuple(v) for k, v in raw.items()})
        rospy.loginfo(f"Goals loaded: {TASK_POSITIONS}")

    # 그리드 로드 및 설정
    grid = load_grid()
    down = rospy.get_param('~downsample_factor', 1)
    GRID_CELL_SIZE = MAP_RESOLUTION * down
    cx = rospy.get_param('/map_center_x')
    cy = rospy.get_param('/map_center_y')
    GRID_ORIGIN_X = cx - (grid.shape[1] * GRID_CELL_SIZE) / 2.0
    GRID_ORIGIN_Y = cy - (grid.shape[0] * GRID_CELL_SIZE) / 2.0

    # TF 리스너 초기화
    tf_listener = tf.TransformListener()
    
    # 로봇별 퍼블리셔 및 구독자 설정
    robots = ['robot_1', 'robot_2']
    for robot in robots:
        pub = rospy.Publisher(f"/{robot}/path", Path, queue_size=1)
        publishers[robot] = pub
        rospy.Subscriber(f"/{robot}/goal", String, goal_callback, 
                         (robot, tf_listener, grid, pub, robots))

    # 경로 업데이트 타이머 설정
    rospy.Timer(rospy.Duration(PATH_UPDATE_RATE), lambda event: update_paths())

    rospy.loginfo("Multi-robot simultaneous path planner ready")
    rospy.loginfo(f"Goal collection timeout: {GOAL_COLLECTION['collection_timeout']} seconds")
    rospy.loginfo(f"Path update rate: {PATH_UPDATE_RATE} seconds per step")
    rospy.spin()
