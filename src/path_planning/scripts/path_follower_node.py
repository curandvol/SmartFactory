#!/usr/bin/env python3
import rospy, math
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
import tf_conversions

class PathFollower:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.cmd_pub = rospy.Publisher(f"/{robot_id}/cmd_vel", Twist, queue_size=10)

        self.current_path = []
        self.current_index = 0
        self.x = self.y = self.theta = None

        rospy.Subscriber(f"/{robot_id}/path", Path, self.path_callback)
        rospy.Subscriber(f"/{robot_id}/odom", Odometry, self.odom_callback)
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        rospy.loginfo(f"[{robot_id}] Path follower ready.")

    def path_callback(self, msg):
        self.current_path = msg.poses
        self.current_index = 0
        rospy.loginfo(f"[{self.robot_id}] Received path with {len(self.current_path)} poses")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.theta = tf_conversions.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])

    def timer_callback(self, event):
        if self.x is None or not self.current_path or self.current_index >= len(self.current_path):
            rospy.loginfo(f"[{self.robot_id}] 정지 조건 - 위치 없음 또는 경로 없음 또는 끝 도달")
            self.publish_cmd(0.0, 0.0)
            return

        target = self.current_path[self.current_index].pose.position
        dx, dy = target.x - self.x, target.y - self.y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        err_yaw = math.atan2(math.sin(target_yaw - self.theta),
                             math.cos(target_yaw - self.theta))

        rospy.loginfo(f"[{self.robot_id}] 현재 인덱스 {self.current_index}/{len(self.current_path)}, dist={dist:.3f}")

        if dist < 0.05:  # 너무 가까우면 다음으로 넘김
            rospy.loginfo(f"[{self.robot_id}] 가까움 → 다음 인덱스로")
            self.current_index += 1
            return

        lin = max(min(0.3 * dist, 0.5), -0.5)
        ang = max(min(1.0 * err_yaw, 1.0), -1.0)
        rospy.loginfo(f"[{self.robot_id}] ➡️ lin={lin:.3f}, ang={ang:.3f}")
        self.publish_cmd(lin, ang)

    def publish_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('path_follower_node')
    robot_id = rospy.get_param('~robot_id', 'robot_1')
    PathFollower(robot_id)
    rospy.spin()

