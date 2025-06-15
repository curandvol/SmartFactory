#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
import tf2_ros
import math
import tf_conversions

class PathFollower:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.cmd_pub = rospy.Publisher(f"/{robot_id}/cmd_vel", Twist, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.current_path = []
        self.current_index = 0
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        rospy.Subscriber(f"/{robot_id}/path", Path, self.path_callback)
        self.x, self.y, self.theta = 0.0, 0.0, 0.0

        rospy.loginfo(f"[{robot_id}] Path follower ready.")

    def path_callback(self, msg):
        self.current_path = msg.poses
        self.current_index = 0
        rospy.loginfo(f"[{self.robot_id}] Received path with {len(self.current_path)} poses")

    def timer_callback(self, event):
        if not self.current_path or self.current_index >= len(self.current_path):
            self.publish_cmd(0.0, 0.0)
            return

        target_pose = self.current_path[self.current_index].pose
        dx = target_pose.position.x - self.x
        dy = target_pose.position.y - self.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < 0.05:
            self.current_index += 1
        else:
            angle = math.atan2(dy, dx)
            self.x += 0.05 * math.cos(angle)
            self.y += 0.05 * math.sin(angle)
            self.theta = angle
            self.publish_cmd(0.2, 0.0)

        # TF 브로드캐스트
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = f"{self.robot_id}/odom"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        quat = tf_conversions.transformations.quaternion_from_euler(0, 0, self.theta)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_cmd(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    rospy.init_node('path_follower_node')
    robot_id = rospy.get_param('~robot_id', 'robot_1')
    PathFollower(robot_id)
    rospy.spin()
