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

TASK_POSITIONS = {
    'goal1': (-3.0, -1.5),
    'goal2': (-2.5, -1.0),
}

MAP_ORIGIN = (-3.666302, -1.963480)  # .yaml 파일과 동일
MAP_RESOLUTION = 0.05  # m/pixel

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, end):
    H, W = grid.shape
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    came_from = {}
    pq = [(f_score[start], start)]

    while pq:
        _, current = heapq.heappop(pq)
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dr, dc in moves:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(pq, (f_score[neighbor], neighbor))
    return None

def world_to_grid(x, y):
    col = int((x - MAP_ORIGIN[0]) / MAP_RESOLUTION)
    row = int((y - MAP_ORIGIN[1]) / MAP_RESOLUTION)
    # 2) NumPy 인덱스(row=0이 최상단)이므로 아래로 뒤집기
    row = grid.shape[0] - 1 - row
    # 3) 클램핑
    row = max(0, min(row, grid.shape[0]-1))
    col = max(0, min(col, grid.shape[1]-1))
    return row, col


def grid_to_world(r, c):
# 2) NumPy 인덱스(row=0이 최상단)이므로 아래로 뒤집기
    row = grid.shape[0] - 1 - r
    # 3) 클램핑
    row = max(0, min(row, grid.shape[0]-1))
    col = max(0, min(c, grid.shape[1]-1))
    
    x = MAP_ORIGIN[0] + (col + 0.5) * MAP_RESOLUTION
    y = MAP_ORIGIN[1] + (row + 0.5) * MAP_RESOLUTION
    rospy.loginfo(f"[DEBUG grid_to_world] Grid (row={r}, col={c}) -> World ({x:.2f}, {y:.2f})")
    return x, y

def path_to_PathMsg(path, frame_id='map'):
    msg = Path()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    for r, c in path:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x, pose.pose.position.y = grid_to_world(r, c)
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)

    rospy.loginfo(f"[DEBUG] Publishing path: frame_id={msg.header.frame_id}, num poses={len(msg.poses)}")
    for i, pose in enumerate(msg.poses[:5]):
        rospy.loginfo(f"Pose {i}: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}")

    return msg

def load_grid():
    # 1) 런타임에 path_planning 패키지 경로를 얻어서,
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('path_planning')

    # 2) 그 아래 maps/grid.npy 를 기본값으로 삼고,
    default_grid = os.path.join(pkg_path, 'maps', 'grid.npy')

    # 3) launch 파일에서 ~grid_path 파라미터가 넘어오면 그걸, 아니면 default_grid 를 사용
    grid_path = rospy.get_param("~grid_path", default_grid)

    while not os.path.exists(grid_path):
        rospy.logwarn(f"Waiting for {grid_path} to be generated...")
        rospy.sleep(0.5)
        
    grid = np.load(grid_path)
    rospy.loginfo(f"Loaded grid from {grid_path}, shape: {grid.shape}")
    return grid

def find_current_grid(tf_listener, robot_frame):
    try:
        now = rospy.Time(0)
        tf_listener.waitForTransform("map", robot_frame, now, rospy.Duration(1.0))
        (trans, _) = tf_listener.lookupTransform("map", robot_frame, now)
        rospy.loginfo(f"[DEBUG TF] {robot_frame} at map: ({trans[0]:.2f}, {trans[1]:.2f})")
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

    path = astar(grid_dyn, start, goal)
    if path is None:
        rospy.logerr(f"[{robot_name}] No path found from {start} to {goal}")
    else:
        path_msg = path_to_PathMsg(path)
        pub.publish(path_msg)

if __name__ == '__main__':
    rospy.init_node('path_planner_node')
    if rospy.has_param('goals'):
    # /goals 파라미터로 넘어온 dict 로 기존 TASK_POSITIONS 를 업데이트
        raw_goals = rospy.get_param('goals')
        TASK_POSITIONS.clear()
        # 리스트→튜플 변환해 주는 게 좋습니다
        for name, coords in raw_goals.items():
            TASK_POSITIONS[name] = tuple(coords)
        rospy.loginfo(f"[path_planner_node] Loaded goals: {TASK_POSITIONS}")
    else:
        rospy.logwarn("[path_planner_node] No 'goals' param found, using defaults.")
        
    tf_listener = tf.TransformListener()
    grid = load_grid()
    robots = ['robot_1', 'robot_2']
    for robot in robots:
        pub = rospy.Publisher(f"/{robot}/path", Path, queue_size=10)
        rospy.Subscriber(f"/{robot}/goal", String, goal_callback, (robot, tf_listener, grid, pub, robots))

    rospy.loginfo("[path_planner_node] Ready for goal commands.")
    rospy.spin()
