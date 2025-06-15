#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import heapq
import os
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


# === 사전 정의된 목적지 (SLAM map 좌표 기준) === #
TASK_POSITIONS = {
    'task1': (0.165, 0.116),
    'task2': (0.317, 9.369),
    'task3': (11.622, 9.344),
    'end':   (-2.108, 4.960),
}

# === 맵 관련 파라미터 === #
MAP_RESOLUTION = 0.05
MAP_ORIGIN = (-10.0, -10.0)
GRID_H, GRID_W = 50, 50

# === A* 알고리즘 === #
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, end):
    H, W = grid.shape
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
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
    gx = int((x - MAP_ORIGIN[0]) / (MAP_RESOLUTION * GRID_W))
    gy = int((y - MAP_ORIGIN[1]) / (MAP_RESOLUTION * GRID_H))
    return gy, gx

def grid_to_world(r, c):
    x = c * (MAP_RESOLUTION * GRID_W) + (MAP_RESOLUTION * GRID_W) / 2 + MAP_ORIGIN[0]
    y = r * (MAP_RESOLUTION * GRID_H) + (MAP_RESOLUTION * GRID_H) / 2 + MAP_ORIGIN[1]
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
    return msg

def load_grid():
    path = rospy.get_param("~grid_path", "/home/yujin/smart_factory_ws/src/path_planning/maps/grid.npy")
    while not os.path.exists(path):
        rospy.logwarn(f"Waiting for {path} to be generated...")
        rospy.sleep(0.5)
    return np.load(path)

def find_current_grid(tf_listener, robot_frame):
    try:
        now = rospy.Time(0)
        tf_listener.waitForTransform("map", robot_frame, now, rospy.Duration(1.0))
        (trans, _) = tf_listener.lookupTransform("map", robot_frame, now)
        return world_to_grid(trans[0], trans[1])
    except Exception as e:
        rospy.logwarn(f"TF lookup failed for {robot_frame}: {e}")
        return None

def task_callback(msg, args):
    robot_name, tf_listener, grid, pub = args
    task_name = msg.data.strip()
    if task_name not in TASK_POSITIONS:
        rospy.logerr(f"Unknown task: {task_name}")
        return

    goal_pos = TASK_POSITIONS[task_name]
    goal = world_to_grid(*goal_pos)
    start = find_current_grid(tf_listener, f"{robot_name}/base_link")

    if not start or grid[start[0]][start[1]] != 0:
        rospy.logerr(f"Invalid start position for {robot_name}: {start}")
        return

    if grid[goal[0]][goal[1]] != 0:
        rospy.logwarn(f"Goal {task_name} is blocked. Finding nearest free cell...")
        goal = find_nearest_free(grid, goal)

    path = astar(grid, start, goal)
    if path is None:
        rospy.logerr(f"[{robot_name}] No path found from {start} to {goal}")
    else:
        path_msg = path_to_PathMsg(path)
        pub.publish(path_msg)
        rospy.loginfo(f"[{robot_name}] Path to {task_name} published. Length: {len(path)}")

def find_nearest_free(grid, pos):
    from collections import deque
    visited = set()
    q = deque([pos])
    H, W = grid.shape
    while q:
        r, c = q.popleft()
        if not (0 <= r < H and 0 <= c < W) or (r, c) in visited:
            continue
        visited.add((r, c))
        if grid[r][c] == 0:
            return (r, c)
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            q.append((r+dr, c+dc))
    return None

if __name__ == '__main__':
    rospy.init_node('path_planner_node')
    tf_listener = tf.TransformListener()

    grid = load_grid()

    robots = ['robot_1', 'robot_2']
    for robot in robots:
        pub = rospy.Publisher(f"/{robot}/path", Path, queue_size=1)
        rospy.Subscriber(f"/{robot}/assign_task", String, task_callback, (robot, tf_listener, grid, pub))

    rospy.loginfo("[path_planner_node] Ready to assign tasks.")
    rospy.spin()
