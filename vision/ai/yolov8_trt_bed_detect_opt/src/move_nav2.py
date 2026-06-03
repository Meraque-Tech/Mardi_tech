#! /usr/bin/env python3
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock
from std_msgs.msg import UInt8
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger
from time import sleep
import os 
import sys

class Goal_pose_nav(Node):
    def __init__(self):
        super().__init__('Goal_pose_nav')

        self.pose_array_sub = self.create_subscription(Path, 'pose_array', self.pose_array_cb, 10)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.cli = self.create_client(Trigger, 'bed_algo_loop')
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('bed_algo_loop service not available, waiting again...')
        # self._action_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        self.req = Trigger.Request()
        

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def pose_array_cb(self, msg):
        print("font len of the bed_pose array --> ", len(msg.poses))
        self.ic = 0
        self.poses_len = len(msg.poses)
        self.send_multiple_goal(msg.poses[0])
        self.send_request()
        

    
    def send_multiple_goal(self, pointStamp):
        goal_poses = NavigateToPose.Goal()
        goal_poses.pose = pointStamp

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Action server detected.')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_poses,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
    
    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded! '+str(self.ic))
            self.ic +=1
            if self.ic<self.poses_len:
                self.send_multiple_goal(msg.poses[self.ic])

        elif status == GoalStatus.STATUS_ABORTED:
            print("goal is Aborted")
            # self.ic +=1
        else:
            self.get_logger().info(
                'Navigation failed with status: {0}'.format(status))




def main(args=None):
    rclpy.init(args=args)
    goal_pose_nav = Goal_pose_nav()
    rclpy.spin(goal_pose_nav)
    goal_pose_nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()